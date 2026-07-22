"""
ott_resolver.py — offline species-name → OTT id resolution against the local dump.

This is the *resolution* half of taxon handling (recognition — finding names in
raw text — is done upstream by the NER head or gnfinder). It maps a candidate
name string to a canonical Open Tree Taxonomy concept, robustly, using a cascade:

    1. normalize    deterministic cleanup: case, unicode, authorship, rank tags,
                    punctuation, whitespace.
    2. exact        hit on a preferred term OR a synonym. OTT synonyms already
                    include cross-database variants ("Haemophilus parasuis") and
                    abbreviated forms ("G. parasuis"), so most "written
                    differently" cases resolve here with no fuzziness.
    3. abbrev       "A. gambiae" → expand the genus initial against known epithets.
    4. fuzzy        misspellings ("Triticum aestivm", "Anophelis gambiae") — a
                    prefix-blocked similarity search, deliberately conservative so
                    it never silently swaps in the wrong taxon.

Backed by a prebuilt SQLite index (see build_index), so the 2.4 GB JSON is parsed
once, offline, and never touched at query time. Returns OTT ids so the rest of the
pipeline (ampliseq_rerank, KG nodes) stays on a single backbone.

Build once:
    python -m src.data.ott_resolver build \
        --json classifier/data/taxonomies/ott_v3.7.2.json \
        --db   classifier/data/processed/ott_index.sqlite

Query:
    from src.data.ott_resolver import OTTResolver
    r = OTTResolver()                       # opens the prebuilt sqlite (read-only)
    hit = r.resolve("Anophelis gambiae")    # -> OTTHit(ott_id='ott:...', ...)
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import threading
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional

# rapidfuzz is a faster, optional drop-in for the ratio; difflib (stdlib) is the
# fallback so the resolver has no hard third-party dependency.
try:
    from rapidfuzz.fuzz import ratio as _rf_ratio  # type: ignore

    def _similarity(a: str, b: str) -> float:
        return _rf_ratio(a, b) / 100.0
except Exception:  # pragma: no cover - exercised only when rapidfuzz absent
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()


_DEFAULT_DB = (
    Path(__file__).resolve().parents[2] / "data/processed/ott_index.sqlite"
)

# Kingdom-rank OTT concepts are climbed to via `parent`; the resolver walks up
# until it hits one of these rank names and reports that concept's term.
_KINGDOM_RANKS = {"kingdom", "domain", "superkingdom"}

# Fuzzy guards: short names are dangerous (one edit flips species), so require a
# high similarity and a minimum length before accepting a fuzzy match.
_FUZZY_MIN_SIM = 0.90
_FUZZY_MIN_LEN = 8
_FUZZY_MAX_CANDIDATES = 400


# ── Normalization ──────────────────────────────────────────────────────────────

# Trailing authorship tokens are stripped *before* lower-casing, because they are
# identified by their capitalisation — authorities are capitalised ("L.", "Merr.")
# whereas epithets are not, so a trailing capitalised token is authorship, never
# part of a binomial. Applied repeatedly to peel e.g. "(L.) Merr. 1917".
_TRAIL_PAREN = re.compile(r"\s*\([^()]*\)\s*$")            # "(L.)"
_TRAIL_YEAR = re.compile(r"\s*,?\s*\d{4}\s*$")             # ", 1753"
_TRAIL_AUTHOR = re.compile(r"\s+[A-Z][A-Za-z'’.\-]*\.?$")  # " Merr." / " L."
# Lowercase nobiliary particles left dangling once the capitalised surname is
# peeled, e.g. "van Tieghem" → "van" → gone.
_TRAIL_PARTICLE = re.compile(
    r"\s+(?:van|von|de[nrl]?|del|della|di|da|la|le|el|du|dos|das|bin|ibn|ter|ten)$",
    re.IGNORECASE,
)
_RANK_TAGS = re.compile(
    r"\b(?:subsp|ssp|var|cf|aff|sp|spp|nr|gen|f)\.?\b", re.IGNORECASE
)
_PUNCT = re.compile(r"[^a-z0-9\s.]")
_WS = re.compile(r"\s+")


def _strip_authorship(s: str) -> str:
    """Peel trailing parentheticals, years, and capitalised authority tokens."""
    prev = None
    while s != prev:
        prev = s
        s = _TRAIL_PAREN.sub("", s)
        s = _TRAIL_YEAR.sub("", s)
        s = _TRAIL_AUTHOR.sub("", s)
        s = _TRAIL_PARTICLE.sub("", s)
    return s.strip()


def normalize_name(name: str) -> str:
    """Deterministic name cleanup used as the SQLite key.

    Strips accents and trailing authorship (on original case), then lower-cases,
    removes rank qualifiers and stray punctuation, and collapses whitespace.
    Lossless enough that a clean binomial maps to itself.

    >>> normalize_name("Quercus robur L.")
    'quercus robur'
    >>> normalize_name("Bellis perennis (L.) 1753")
    'bellis perennis'
    """
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _strip_authorship(s.strip())
    s = s.lower()
    s = _RANK_TAGS.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _prefix3(name_norm: str) -> str:
    """Blocking key for the fuzzy pass: first 3 alnum chars of the genus."""
    head = name_norm.replace(" ", "")
    return head[:3]


# ── Query-side resolver ────────────────────────────────────────────────────────

@dataclass
class OTTHit:
    ott_id: str            # "ott:1047118"
    canonical: str         # preferred term of the resolved concept
    rank: str              # "species", "genus", ...
    kingdom: Optional[str]  # climbed via parents, when resolvable
    match_type: str        # "exact" | "synonym" | "abbrev" | "fuzzy"
    score: float           # 1.0 for deterministic hits; similarity for fuzzy
    query: str


class OTTResolver:
    """Read-only resolver over the prebuilt SQLite index."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or _DEFAULT_DB)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"OTT index not found at {self.db_path}. Build it once with:\n"
                f"    python -m src.data.ott_resolver build --json <ott.json> "
                f"--db {self.db_path}"
            )
        # The service serves sync endpoints from FastAPI's threadpool, so a single
        # shared sqlite connection would raise "objects created in a thread can only
        # be used in that same thread". Give each thread its own read-only
        # connection, opened lazily (read-only + separate connections ⇒ no lock).
        self._local = threading.local()
        self._has_fuzzy = self._table_exists("names") and self._column_exists(
            "names", "prefix3"
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
        )

    @property
    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = self._local.con = self._connect()
        return con

    # -- public API ------------------------------------------------------------

    def resolve(self, name: str, fuzzy: bool = True) -> Optional[OTTHit]:
        """Resolve one name → OTTHit, or None if unresolved."""
        norm = normalize_name(name)
        if not norm:
            return None
        return (
            self._exact(norm)
            or self._abbrev(norm)
            or (self._fuzzy(norm) if fuzzy else None)
        )

    def resolve_id(self, name: str, fuzzy: bool = True) -> Optional[str]:
        """Convenience: return just the 'ott:ID' string (drop-in for resolve_ott)."""
        hit = self.resolve(name, fuzzy=fuzzy)
        return hit.ott_id if hit else None

    # -- cascade steps ---------------------------------------------------------

    def _exact(self, norm: str) -> Optional[OTTHit]:
        row = self._con.execute(
            "SELECT ott_id, is_syn FROM names WHERE name_norm = ? "
            "ORDER BY is_syn ASC LIMIT 1",
            (norm,),
        ).fetchone()
        if row:
            return self._build_hit(
                row[0], norm, "synonym" if row[1] else "exact", 1.0
            )
        return None

    def _abbrev(self, norm: str) -> Optional[OTTHit]:
        # "a. gambiae" → genus initial + epithet. Expand against known epithets.
        m = re.match(r"^([a-z])\.?\s+([a-z][a-z-]+)$", norm)
        if not m:
            return None
        initial, epithet = m.group(1), m.group(2)
        rows = self._con.execute(
            "SELECT DISTINCT ott_id, name_norm FROM names "
            "WHERE name_norm LIKE ? AND name_norm LIKE ?",
            (f"{initial}% {epithet}", f"% {epithet}"),
        ).fetchall()
        # Accept only when the expansion is unambiguous.
        uniq = {r[0] for r in rows}
        if len(uniq) == 1:
            ott_id = rows[0][0]
            return self._build_hit(ott_id, norm, "abbrev", 1.0)
        return None

    def _fuzzy(self, norm: str) -> Optional[OTTHit]:
        if not self._has_fuzzy or len(norm) < _FUZZY_MIN_LEN:
            return None
        cands = self._con.execute(
            "SELECT name_norm, ott_id, is_syn FROM names WHERE prefix3 = ? "
            "LIMIT ?",
            (_prefix3(norm), _FUZZY_MAX_CANDIDATES),
        ).fetchall()
        best = None
        best_sim = 0.0
        for cand_norm, ott_id, is_syn in cands:
            sim = _similarity(norm, cand_norm)
            if sim > best_sim:
                best_sim, best = sim, (ott_id, cand_norm)
        if best and best_sim >= _FUZZY_MIN_SIM:
            return self._build_hit(best[0], norm, "fuzzy", round(best_sim, 3))
        return None

    # -- helpers ---------------------------------------------------------------

    def _build_hit(
        self, ott_id: str, query: str, match_type: str, score: float
    ) -> Optional[OTTHit]:
        row = self._con.execute(
            "SELECT pref, rank FROM concepts WHERE ott_id = ?", (ott_id,)
        ).fetchone()
        if not row:
            return None
        pref, rank = row
        return OTTHit(
            ott_id=f"ott:{ott_id}",
            canonical=pref,
            rank=rank or "",
            kingdom=self._kingdom_of(ott_id),
            match_type=match_type,
            score=score,
            query=query,
        )

    def _kingdom_of(self, ott_id: str, _max_depth: int = 40) -> Optional[str]:
        cur = ott_id
        for _ in range(_max_depth):
            row = self._con.execute(
                "SELECT pref, rank, parent FROM concepts WHERE ott_id = ?", (cur,)
            ).fetchone()
            if not row:
                return None
            pref, rank, parent = row
            if (rank or "").lower() in _KINGDOM_RANKS:
                return pref
            if not parent:
                return None
            cur = parent
        return None

    def _table_exists(self, name: str) -> bool:
        return bool(
            self._con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ).fetchone()
        )

    def _column_exists(self, table: str, col: str) -> bool:
        cols = [r[1] for r in self._con.execute(f"PRAGMA table_info({table})")]
        return col in cols


# ── Index builder (one-off, offline) ───────────────────────────────────────────

def _iter_concepts(json_path: Path) -> Iterable[dict]:
    """Stream concepts from the OTT dump. Uses ijson if available (low memory),
    else falls back to a full json.load (needs plenty of RAM for the 2.4 GB file).
    """
    try:
        import ijson  # type: ignore

        with open(json_path, "rb") as fh:
            yield from ijson.items(fh, "concepts.item")
    except ImportError:
        import json

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        yield from data.get("concepts", [])


def build_index(
    json_path: Path, db_path: Path, with_fuzzy: bool = True
) -> None:
    """Parse the OTT JSON once into a queryable SQLite index."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.executescript(
        """
        CREATE TABLE concepts (ott_id TEXT PRIMARY KEY, pref TEXT, rank TEXT, parent TEXT);
        CREATE TABLE names (name_norm TEXT, ott_id TEXT, is_syn INTEGER, prefix3 TEXT);
        """
    )

    n_concepts = n_names = 0
    concept_rows, name_rows = [], []
    for c in _iter_concepts(json_path):
        ott_id = str(c.get("id", "")).strip()
        if not ott_id:
            continue
        pref_term = (c.get("preferred_term") or {}).get("term", "")
        rank = (c.get("rank") or {}).get("name", "")
        parents = c.get("parents") or [""]
        parent = str(parents[0]).strip() if parents else ""
        concept_rows.append((ott_id, pref_term, rank, parent))

        # index the preferred term + every synonym
        terms = [(pref_term, 0)] + [
            (s.get("term", ""), 1) for s in (c.get("synonyms") or [])
        ]
        for term, is_syn in terms:
            norm = normalize_name(term)
            if norm:
                name_rows.append((norm, ott_id, is_syn, _prefix3(norm)))
                n_names += 1

        n_concepts += 1
        if len(concept_rows) >= 50_000:
            con.executemany("INSERT INTO concepts VALUES (?,?,?,?)", concept_rows)
            con.executemany("INSERT INTO names VALUES (?,?,?,?)", name_rows)
            con.commit()
            concept_rows, name_rows = [], []
            print(f"  ...{n_concepts:,} concepts, {n_names:,} names", flush=True)

    if concept_rows:
        con.executemany("INSERT INTO concepts VALUES (?,?,?,?)", concept_rows)
    if name_rows:
        con.executemany("INSERT INTO names VALUES (?,?,?,?)", name_rows)
    con.commit()

    print("  building indexes ...", flush=True)
    con.execute("CREATE INDEX ix_names_norm ON names(name_norm)")
    if with_fuzzy:
        con.execute("CREATE INDEX ix_names_prefix ON names(prefix3)")
    con.commit()
    con.close()
    print(
        f"Done: {n_concepts:,} concepts, {n_names:,} names → {db_path}", flush=True
    )


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the SQLite index from the OTT JSON")
    b.add_argument("--json", required=True, type=Path)
    b.add_argument("--db", default=_DEFAULT_DB, type=Path)
    b.add_argument("--no-fuzzy", action="store_true", help="skip the prefix index")

    q = sub.add_parser("resolve", help="resolve names from the command line")
    q.add_argument("names", nargs="+")
    q.add_argument("--db", default=_DEFAULT_DB, type=Path)

    args = ap.parse_args()
    if args.cmd == "build":
        build_index(args.json, args.db, with_fuzzy=not args.no_fuzzy)
    else:
        r = OTTResolver(args.db)
        for name in args.names:
            hit = r.resolve(name)
            if hit:
                print(
                    f"{name!r:40} → {hit.ott_id}  {hit.canonical!r} "
                    f"[{hit.rank}, {hit.kingdom}] ({hit.match_type} {hit.score})"
                )
            else:
                print(f"{name!r:40} → UNRESOLVED")


if __name__ == "__main__":
    _cli()
