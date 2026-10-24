#!/usr/bin/env sh
# POSIX helper.  Usage:  ./run.sh <setup|fetch|pipeline|refit|test|app|e2e|all>
set -e
cd "$(dirname "$0")"
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi

case "${1:-help}" in
  setup)    [ -x "$PY" ] || python3 -m venv .venv
            if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
            "$PY" -m pip install --upgrade pip && "$PY" -m pip install -r requirements.txt && "$PY" -m pip install -e . --no-deps ;;
  fetch)    "$PY" scripts/fetch_supplementary_asvspoof.py ;;
  pipeline) "$PY" -m voicedetect.pipeline all ;;
  refit)    "$PY" scripts/train.py --refit ;;
  test)     "$PY" -m pytest ;;
  app)      "$PY" -m streamlit run app/streamlit_app.py --server.port 8501 ;;
  e2e)      "$PY" scripts/e2e_check.py ;;
  all)      "$0" setup && "$0" fetch && "$0" pipeline && "$0" test ;;
  *)        echo "Usage: ./run.sh <setup|fetch|pipeline|refit|test|app|e2e|all>" ;;
esac
