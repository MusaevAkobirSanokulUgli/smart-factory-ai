#!/usr/bin/env bash
# Smart-Factory AI — one-click launcher (bash / git-bash on Windows)
set -euo pipefail

VENV_PY="D:/Ai_Portfolio/.venv/Scripts/python.exe"
APP_DIR="$(cd "$(dirname "$0")" && pwd)/backend"

if [[ ! -f "$VENV_PY" ]]; then
  echo "[ERROR] Python venv not found at $VENV_PY"
  echo "        Make sure D:/Ai_Portfolio/.venv exists (see Ai_Portfolio/SETUP.md)."
  exit 1
fi

echo "----------------------------------------------------------"
echo " Smart-Factory AI — Unified Showcase App"
echo " Serving on http://127.0.0.1:7860/"
echo " OpenAPI  at http://127.0.0.1:7860/docs"
echo " Press Ctrl+C to stop."
echo "----------------------------------------------------------"

cd "$APP_DIR"
"$VENV_PY" -m uvicorn main:app --host 127.0.0.1 --port 7860
