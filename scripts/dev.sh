#!/usr/bin/env bash
# Starts the API and the dashboard together. Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "Ollama is not responding on :11434. Start it with 'ollama serve'." >&2
  exit 1
fi

if [ ! -x backend/.venv/bin/python ]; then
  echo "No venv. Run: cd backend && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

trap 'kill 0' EXIT
backend/.venv/bin/python -m uvicorn app.main:app --app-dir backend --port 8000 --reload &
(cd frontend && npm run dev) &
wait
