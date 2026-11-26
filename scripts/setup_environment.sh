#!/usr/bin/env bash
# Setup for local / Colab / Kaggle T4.
# Usage:
#   bash scripts/setup_environment.sh
#   bash scripts/setup_environment.sh --colab
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="local"
for arg in "$@"; do
  case "$arg" in
    --colab|--kaggle) MODE="cloud" ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo "=== Setup (${MODE}) ROOT=${ROOT} ==="

if [[ "$MODE" == "local" ]]; then
  if [[ ! -d "$ROOT/.venv" ]]; then
    python3 -m venv "$ROOT/.venv"
  fi
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"

echo ""
echo "=== Verifying environment + data yaml path ==="
python "$ROOT/scripts/check_environment.py"

echo ""
echo "Next:"
echo "  1. Place RDD2022 under data/raw/RDD2022/ (see data/README.md)"
echo "  2. Ensure data/splits/*.csv exist (tracked in repo)"
echo "  3. Ensure processed YOLO layout exists OR run conversion scripts"
echo "  4. python scripts/reproduce_main_result.py"
