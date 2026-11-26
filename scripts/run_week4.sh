#!/usr/bin/env bash
# Week 4 pipeline:
#   0) Verify Week 3 outputs
#   1) Train YOLO11n (same split / seed / budget family as YOLOv8n)
#   2) Compare YOLOv8n vs YOLO11n
#   3) Evaluate both on val + test (test never used for selection)
#   4) Write run_manifest.json
#
# Usage:
#   ./scripts/run_week4.sh
#   ./scripts/run_week4.sh --smoke
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${ALLOW_LEGACY_FIXTURE:-0}" != "1" ]]; then
  echo "LEGACY synthetic-fixture runner (Week 4). Official training:" >&2
  echo "  python scripts/train_yolo11_baseline.py --config configs/rdd2022.yaml" >&2
  echo "Set ALLOW_LEGACY_FIXTURE=1 to run it anyway (debugging only)." >&2
  exit 2
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-6845}"
SMOKE=0
TRAIN_EXTRA=()
COMPARE_EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --smoke)
      SMOKE=1
      TRAIN_EXTRA+=(--smoke)
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Supported: --smoke" >&2
      exit 2
      ;;
  esac
done

STARTED="$("$PYTHON" - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

echo "============================================================"
echo "Week 4 pipeline"
echo "  ROOT=$ROOT"
echo "  SEED=$SEED"
echo "  SMOKE=$SMOKE"
echo "  STARTED=$STARTED"
echo "============================================================"

echo ""
echo "[0/4] Verify Week 3 outputs ..."
missing=0
for f in \
  configs/rdd2022.yaml \
  data/splits/train.csv \
  data/splits/val.csv \
  data/splits/test.csv \
  data/splits/split_summary.json \
  results/week3/yolov8n_baseline/metrics.json
do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f" >&2
    missing=1
  else
    echo "  OK: $f"
  fi
done
# Checkpoint path inside metrics JSON
if ! "$PYTHON" - <<'PY'
import json, sys
from pathlib import Path
card = json.loads(Path("results/week3/yolov8n_baseline/metrics.json").read_text())
ckpt = card.get("best_checkpoint")
ok = isinstance(ckpt, str) and Path(ckpt).is_file()
print("  OK: YOLOv8n checkpoint" if ok else f"  MISSING checkpoint: {ckpt}")
sys.exit(0 if ok else 1)
PY
then
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  echo "ERROR: Week 3 outputs incomplete. Run ./scripts/run_week3.sh first." >&2
  exit 1
fi

echo ""
echo "[1/4] Train YOLO11n strong baseline ..."
"$PYTHON" scripts/train_yolo11_baseline.py \
  --data configs/rdd2022.yaml \
  --seed "$SEED" \
  "${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}"

echo ""
echo "[2/4] Compare YOLOv8n vs YOLO11n ..."
"$PYTHON" scripts/compare_baselines.py \
  --seed "$SEED" \
  "${COMPARE_EXTRA[@]+"${COMPARE_EXTRA[@]}"}"

echo ""
echo "[3/4] Evaluate both models (val + test; selection was val-only) ..."
"$PYTHON" scripts/evaluate_week4.py --seed "$SEED"

echo ""
echo "[4/4] Write run manifest ..."
CMDS=(
  "python scripts/train_yolo11_baseline.py --data configs/rdd2022.yaml --seed ${SEED}"
  "python scripts/compare_baselines.py --seed ${SEED}"
  "python scripts/evaluate_week4.py --seed ${SEED}"
)
if [[ "$SMOKE" -eq 1 ]]; then
  CMDS[0]="python scripts/train_yolo11_baseline.py --data configs/rdd2022.yaml --seed ${SEED} --smoke"
fi
"$PYTHON" scripts/write_week4_manifest.py \
  --started-at "$STARTED" \
  $([[ "$SMOKE" -eq 1 ]] && echo --smoke) \
  --commands "${CMDS[@]}"

echo ""
echo "Week 4 pipeline finished."
echo "  YOLO11n:     results/week4/yolo11n_baseline/"
echo "  Comparison:  results/week4/baseline_comparison.csv"
echo "  Evaluation:  results/week4/evaluation/"
echo "  Figures:     figures/week4/"
echo "  Manifest:    results/week4/run_manifest.json"
echo "  Report:      docs/week4_progress_report.md  (generate/update after runs)"
