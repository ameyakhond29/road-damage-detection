#!/usr/bin/env bash
# Week 5 pipeline — lightweight architecture / resolution ablations.
#
#   0) Verify Week 3/4 split + model YAMLs
#   1) Unit tests for model components
#   2) Efficiency profiling (architecture init or trained ckpts)
#   3) Train each experiment × seeds (0/1/2) — or smoke seed=0 only
#   4) Aggregate val metrics + figures (NO test evaluation)
#   5) Write model_selection_decision.md from val-only summaries
#
# Usage:
#   ./scripts/run_week5.sh
#   ./scripts/run_week5.sh --smoke
#   SEEDS="0 1 2" ./scripts/run_week5.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${ALLOW_LEGACY_FIXTURE:-0}" != "1" ]]; then
  echo "LEGACY synthetic-fixture runner (Week 5). Official training:" >&2
  echo "  python scripts/train_week5_ablation.py --experiment yolo11n_640" >&2
  echo "Set ALLOW_LEGACY_FIXTURE=1 to run it anyway (debugging only)." >&2
  exit 2
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SMOKE=0
TRAIN_EXTRA=()
EFF_EXTRA=()
SEEDS_DEFAULT="0 1 2"

for arg in "$@"; do
  case "$arg" in
    --smoke)
      SMOKE=1
      TRAIN_EXTRA+=(--smoke)
      EFF_EXTRA+=(--smoke)
      SEEDS_DEFAULT="0"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Supported: --smoke" >&2
      exit 2
      ;;
  esac
done

SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
EXPERIMENTS=(
  yolo11n_640
  yolo11n_p2_640
  yolo11n_attention_640
  yolo11n_800
  combined_800
)

STARTED="$("$PYTHON" - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

echo "============================================================"
echo "Week 5 ablation pipeline"
echo "  ROOT=$ROOT"
echo "  SMOKE=$SMOKE"
echo "  SEEDS=$SEEDS"
echo "  STARTED=$STARTED"
echo "============================================================"

echo ""
echo "[0/5] Verify split + model YAMLs ..."
missing=0
for f in \
  configs/rdd2022.yaml \
  data/splits/train.csv \
  data/splits/val.csv \
  data/splits/test.csv \
  data/splits/split_summary.json \
  configs/week5_yolo11n_640.yaml \
  configs/week5_yolo11n_p2_640.yaml \
  configs/week5_yolo11n_attention_640.yaml \
  configs/week5_yolo11n_800.yaml \
  configs/week5_combined_800.yaml \
  configs/models/yolo11n-p2.yaml \
  configs/models/yolo11n-attn.yaml \
  configs/models/yolo11n-p2-attn.yaml
do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f" >&2
    missing=1
  else
    echo "  OK: $f"
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Aborting: required files missing." >&2
  exit 1
fi

# Confirm identical split hashes across configs
"$PYTHON" - <<'PY'
from pathlib import Path
import hashlib, yaml
root = Path(".")
def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
hashes = {s: h(f"data/splits/{s}.csv") for s in ("train","val","test")}
print("  split hashes:", hashes)
data = yaml.safe_load(Path("configs/rdd2022.yaml").read_text())
assert data.get("split_id") == "grouped_70_15_15_seed6845", data.get("split_id")
print("  split_id OK:", data.get("split_id"))
# All week5 configs must point at same data yaml
for name in [
    "week5_yolo11n_640.yaml",
    "week5_yolo11n_p2_640.yaml",
    "week5_yolo11n_attention_640.yaml",
    "week5_yolo11n_800.yaml",
    "week5_combined_800.yaml",
]:
    cfg = yaml.safe_load(Path("configs", name).read_text())
    assert cfg.get("data") == "configs/rdd2022.yaml", (name, cfg.get("data"))
    assert cfg.get("test_used_for_selection") is False
print("  all week5 configs share rdd2022.yaml; test_used_for_selection=false")
PY

echo ""
echo "[1/5] Unit tests for Week 5 model components ..."
"$PYTHON" -m unittest tests.test_week5_models -v

echo ""
echo "[2/5] Efficiency profiling ..."
"$PYTHON" scripts/measure_efficiency.py "${EFF_EXTRA[@]}"

echo ""
echo "[3/5] Train ablations ..."
for exp in "${EXPERIMENTS[@]}"; do
  for seed in $SEEDS; do
    echo "---- train $exp seed=$seed ----"
    "$PYTHON" scripts/train_week5_ablation.py \
      --experiment "$exp" \
      --seed "$seed" \
      "${TRAIN_EXTRA[@]}"
  done
done

echo ""
echo "[4/5] Aggregate validation metrics + figures (no test) ..."
"$PYTHON" scripts/evaluate_week5_ablation.py --recompute-val

echo ""
echo "[5/5] Write model selection decision (val only) ..."
"$PYTHON" scripts/write_week5_decision.py

echo ""
echo "============================================================"
echo "Week 5 complete (or smoke)."
echo "Results: results/week5/"
echo "Figures: figures/week5/"
echo "Report:  docs/week5_ablation_report.md"
echo "Decision: results/week5/model_selection_decision.md"
echo "============================================================"
echo ""
echo "Reproduce (smoke):"
echo "  ./scripts/run_week5.sh --smoke"
echo "Reproduce (full, seeds 0 1 2):"
echo "  ./scripts/run_week5.sh"
echo "Single experiment:"
echo "  python scripts/train_week5_ablation.py --experiment yolo11n_640 --seed 0"
echo "  python scripts/train_week5_ablation.py --experiment yolo11n_p2_640 --seed 0"
echo "  python scripts/train_week5_ablation.py --experiment yolo11n_attention_640 --seed 0"
echo "  python scripts/train_week5_ablation.py --experiment yolo11n_800 --seed 0"
echo "  python scripts/train_week5_ablation.py --experiment combined_800 --seed 0"
echo "  python scripts/measure_efficiency.py"
echo "  python scripts/evaluate_week5_ablation.py --recompute-val"
echo "  python scripts/write_week5_decision.py"
