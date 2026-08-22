#!/usr/bin/env bash
#
# Builds the self-contained Python runtime that ships inside the app.
#
# Users must not need Python installed, so a relocatable CPython from
# python-build-standalone is downloaded and the backend's dependencies are
# installed into it. The result lands in desktop/resources/backend and is
# copied into the bundle by electron-builder.
#
# Run this on the platform you are packaging for -- wheels are platform
# specific, and a macOS venv will not run on Windows.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP="$(dirname "$HERE")"
ROOT="$(dirname "$DESKTOP")"
OUT="$DESKTOP/resources/backend"

PY_VERSION="3.12.8"
RELEASE="20241219"

case "$(uname -s)" in
  Darwin) OS="apple-darwin" ;;
  Linux)  OS="unknown-linux-gnu" ;;
  MINGW*|MSYS*|CYGWIN*) OS="pc-windows-msvc" ;;
  *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  arm64|aarch64) ARCH="aarch64" ;;
  x86_64|amd64)  ARCH="x86_64" ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

# Windows standalone builds are x86_64 only in practice.
if [ "$OS" = "pc-windows-msvc" ]; then ARCH="x86_64"; fi

TRIPLE="${ARCH}-${OS}"
ASSET="cpython-${PY_VERSION}+${RELEASE}-${TRIPLE}-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${RELEASE}/${ASSET}"

echo "==> Target: $TRIPLE (CPython $PY_VERSION)"

rm -rf "$OUT"
mkdir -p "$OUT"

CACHE="$DESKTOP/.cache"
mkdir -p "$CACHE"

if [ ! -f "$CACHE/$ASSET" ]; then
  echo "==> Downloading standalone Python"
  curl -fSL --retry 3 -o "$CACHE/$ASSET" "$URL"
else
  echo "==> Using cached $ASSET"
fi

echo "==> Extracting"
tar -xzf "$CACHE/$ASSET" -C "$OUT"
# The archive contains a top-level `python/` directory, which is what main.js
# expects to find.

if [ "$OS" = "pc-windows-msvc" ]; then
  PYBIN="$OUT/python/python.exe"
else
  PYBIN="$OUT/python/bin/python3"
fi

echo "==> Installing dependencies"
"$PYBIN" -m pip install --upgrade pip --no-warn-script-location >/dev/null
"$PYBIN" -m pip install --no-warn-script-location -r "$ROOT/backend/requirements.txt"

echo "==> Copying backend source"
cp -R "$ROOT/backend/app" "$OUT/app"
cp -R "$ROOT/backend/agents" "$OUT/agents" 2>/dev/null || true

echo "==> Trimming"
# Nothing in the bundle compiles or tests anything, so the build-time tooling
# and caches are dead weight in a download users have to wait for.
"$PYBIN" -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true
find "$OUT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -type d -name "tests" -path "*/site-packages/*" -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -type f -name "*.pyc" -delete 2>/dev/null || true

echo "==> Verifying the bundle imports"
( cd "$OUT" && "$PYBIN" -c "import sys; sys.path.insert(0,'.'); import app.main; print('backend imports OK')" )

echo "==> Done: $(du -sh "$OUT" | cut -f1) at $OUT"
