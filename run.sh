#!/usr/bin/env bash
# Launch gui.py, bootstrapping a local venv on first run so no manual
# activate/install is needed.
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
STAMP="$VENV/.requirements.stamp"

if [ ! -x "$VENV/bin/python" ]; then
  echo "No venv found — creating $VENV ..." >&2
  python3 -m venv "$VENV"
fi

if [ ! -f "$STAMP" ] || [ requirements.txt -nt "$STAMP" ]; then
  echo "Installing dependencies (first run, or requirements.txt changed) ..." >&2
  "$VENV/bin/pip" install -q --disable-pip-version-check -r requirements.txt
  touch "$STAMP"
fi

if ! "$VENV/bin/python" -c "import tkinter" 2>/dev/null; then
  echo "tkinter is missing from this Python install." >&2
  echo "Install it with your system package manager, e.g.:" >&2
  echo "  sudo apt install python3-tk" >&2
  exit 1
fi

exec "$VENV/bin/python" gui.py "$@"
