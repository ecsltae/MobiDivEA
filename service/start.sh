#!/usr/bin/env bash
# Start the metadata QA service (species + locations), bound to all interfaces so
# it is reachable on the LAN (not just localhost).
set -euo pipefail

cd "$(dirname "$0")"

export SPECIES_QA_OTT_DB="${SPECIES_QA_OTT_DB:-/home/egaillac/MetaP/classifier/data/processed/ott_index.sqlite}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
VENV="${VENV:-/home/egaillac/MetaP/MPvenv}"

exec "$VENV/bin/uvicorn" app:app --host "$HOST" --port "$PORT" --workers 1
