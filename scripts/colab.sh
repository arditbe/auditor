#!/usr/bin/env bash
#
# Auditor on Google Colab (or any headless Linux box).
#
#   !bash scripts/colab.sh
#
# Installs what is needed, takes a Gemini API key, and runs an audit in the
# terminal. No browser, no dashboard, no Google Cloud project.
#
# Optional environment variables (set these to skip the prompts):
#   GOOGLE_API_KEY   Gemini key from https://aistudio.google.com/apikey
#   TARGET           model to audit, e.g. ollama:qwen2:0.5b or an https:// URL
#   PROBES           how many questions (default 6)
#   SUITE            general | medical | code | safety
#   PURPOSE          what the model was fine-tuned to do
#   WITH_OLLAMA      1 to install Ollama and pull a small model to audit
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'; RESET=$'\033[0m'
say()  { printf '%s\n' "${BOLD}==>${RESET} $*"; }
note() { printf '%s\n' "${DIM}    $*${RESET}"; }
die()  { printf '%s\n' "${RED}error:${RESET} $*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 1. Python
# --------------------------------------------------------------------------

PY=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d%02d" % sys.version_info[:2])')"
    # Google ADK needs 3.10+. Colab ships 3.11, which is fine.
    if [ "$version" -ge 310 ]; then PY="$candidate"; break; fi
  fi
done
[ -n "$PY" ] || die "Python 3.10 or newer is required. Found none."
say "Using $("$PY" --version)"

# Colab already runs as root in a disposable container, so a venv adds a layer
# of confusion for no isolation benefit. Elsewhere, keep the venv.
IN_COLAB=0
if [ -d /content ] && [ -n "${COLAB_RELEASE_TAG:-}${COLAB_GPU:-}" ]; then IN_COLAB=1; fi
if [ -f /.dockerenv ] && [ -d /content ]; then IN_COLAB=1; fi

if [ "$IN_COLAB" = "1" ]; then
  say "Google Colab detected — installing into the system environment"
  PIP=("$PY" -m pip install -q)
  RUN=("$PY")
else
  say "Creating a virtual environment at backend/.venv"
  [ -d backend/.venv ] || "$PY" -m venv backend/.venv
  # Absolute: the run step changes directory into backend/ first.
  VENV_PY="$REPO_ROOT/backend/.venv/bin/python"
  PIP=("$VENV_PY" -m pip install -q)
  RUN=("$VENV_PY")
fi

# --------------------------------------------------------------------------
# 2. Dependencies
# --------------------------------------------------------------------------

say "Installing dependencies (a minute or two the first time)"
"${PIP[@]}" --upgrade pip
"${PIP[@]}" -r "$REPO_ROOT/backend/requirements.txt"
note "google-adk, litellm, fastapi and friends"

# --------------------------------------------------------------------------
# 3. Gemini API key
# --------------------------------------------------------------------------

if [ -z "${GOOGLE_API_KEY:-}" ]; then
  echo
  say "Gemini API key"
  note "Free to create at https://aistudio.google.com/apikey"
  note "Nothing is stored: the key lives in this shell only."
  # -s hides the key so it does not end up in the notebook output.
  printf '    Paste your key (input hidden): '
  read -rs GOOGLE_API_KEY || true
  echo
fi

if [ -z "${GOOGLE_API_KEY:-}" ]; then
  VALIDATOR="local-gemma"
  echo
  note "No key given. Falling back to a local Ollama judge."
  note "That needs Ollama running with gemma3:12b pulled."
else
  export GOOGLE_API_KEY
  VALIDATOR="${VALIDATOR:-gemini-flash-key}"
  printf '%s\n' "    ${GREEN}key accepted${RESET} (${#GOOGLE_API_KEY} characters)"
fi

# --------------------------------------------------------------------------
# 4. Something to audit
# --------------------------------------------------------------------------

if [ "${WITH_OLLAMA:-0}" = "1" ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    say "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    say "Starting Ollama"
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 60); do
      curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  say "Pulling a small model to audit"
  ollama pull qwen2:0.5b
  TARGET="${TARGET:-ollama:qwen2:0.5b}"
fi

if [ -z "${TARGET:-}" ]; then
  echo
  say "What should Auditor test?"
  note "An Ollama tag  ->  ollama:qwen2:0.5b"
  note "A served model ->  https://your-model.example.com/v1"
  note "Or re-run with WITH_OLLAMA=1 to install one here."
  printf '    Target: '
  read -r TARGET || true
fi

[ -n "${TARGET:-}" ] || die "No target given. Set TARGET or re-run with WITH_OLLAMA=1."

# --------------------------------------------------------------------------
# 5. Run
# --------------------------------------------------------------------------

echo
say "Auditing ${TARGET}"
echo

cd "$REPO_ROOT/backend"
exec "${RUN[@]}" -m app.cli \
  --target "$TARGET" \
  --validator "$VALIDATOR" \
  --probes "${PROBES:-6}" \
  --suite "${SUITE:-general}" \
  --purpose "${PURPOSE:-}"
