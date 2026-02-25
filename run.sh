#!/bin/bash
# Launch KangahGPS Web
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"
streamlit run "$DIR/app.py"
