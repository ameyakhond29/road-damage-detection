#!/usr/bin/env bash
# Week 6 pipeline — freeze protocol, one-shot test eval, country/error/efficiency, tables.
#
# Usage:
#   ./scripts/run_week6.sh
#   ./scripts/run_week6.sh --smoke
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${ALLOW_LEGACY_FIXTURE:-0}" != "1" ]]; then
  echo "LEGACY synthetic-fixture runner (Week 6 frozen protocol). Not official RDD2022." >&2
  echo "Set ALLOW_LEGACY_FIXTURE=1 to run it anyway (debugging only)." >&2
  exit 2
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SMOKE=0
EXTRA=()

for arg in "$@"; do
  case "$arg" in
    --smoke)
      SMOKE=1
      EXTRA+=(--smoke)
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Supported: --smoke" >&2
      exit 2
      ;;
  esac
done

echo "============================================================"
echo "Week 6 final evaluation"
echo "  ROOT=$ROOT"
echo "  SMOKE=$SMOKE"
echo "============================================================"

echo ""
echo "[1/8] Verify Week 5 outputs + checkpoints ..."
missing=0
for f in \
  results/week5/model_selection_decision.md \
  results/week5/seed_summary.csv \
  results/week5/yolo11n_640/seed_0/train/weights/best.pt \
  results/week3/yolov8n_baseline/train/weights/best.pt \
  results/week4/yolo11n_baseline/train/weights/best.pt \
  configs/rdd2022.yaml \
  data/splits/train.csv \
  data/splits/val.csv \
  data/splits/test.csv \
  configs/final_evaluation.yaml \
  docs/final_evaluation_protocol.md
do
  if [[ ! -f "$f" ]]; then
    echo "  MISSING: $f" >&2
    missing=1
  else
    echo "  OK: $f"
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Aborting." >&2
  exit 1
fi

"$PYTHON" - <<'PY'
from pathlib import Path
import hashlib, yaml
def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
print("  split hashes:", {s: h(f"data/splits/{s}.csv") for s in ("train","val","test")})
proto = yaml.safe_load(Path("configs/final_evaluation.yaml").read_text())
assert proto.get("protocol_frozen") is True
assert proto["final_model"].get("test_used_for_selection") is False
assert Path(proto["final_model"]["checkpoint"]).is_file()
print("  protocol frozen; final ckpt OK; test_used_for_selection=false")
PY

echo ""
echo "[2/8] Verify frozen protocol doc ..."
test -f docs/final_evaluation_protocol.md

echo ""
echo "[3/8] Final held-out evaluation (val conf lock → one-shot test) ..."
"$PYTHON" scripts/run_final_evaluation.py "${EXTRA[@]}"

echo ""
echo "[4/8] Country-level analysis (already inside step 3; refresh plot if needed) ..."
# country_results.csv written by run_final_evaluation.py
test -f results/week6/country_results.csv

echo ""
echo "[5/8] Error analysis ..."
"$PYTHON" scripts/error_analysis.py "${EXTRA[@]}"

echo ""
echo "[6/8] Efficiency benchmark ..."
"$PYTHON" scripts/final_efficiency_benchmark.py "${EXTRA[@]}"

echo ""
echo "[7/8] Final tables + Week 6 report ..."
"$PYTHON" scripts/write_week6_tables.py

echo ""
echo "[8/8] Final run manifest already written; print confirmations ..."
"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path("results/week6")
lock = json.loads((root/"CONFIDENCE_THRESHOLD_LOCKED.json").read_text())
assert lock.get("test_used") is False
assert lock.get("split") == "val"
print("  conf locked on validation only:", lock.get("selected_conf"))
test_done = (root/"TEST_EVALUATION_COMPLETED.json").is_file()
print("  test evaluation completed:", test_done)
man = json.loads((root/"final_run_manifest.json").read_text())
print("  test_used_for_selection:", man.get("test_used_for_selection"))
print("  split hashes:", man.get("split_hashes"))
for f in [
  "final_test_results.csv","final_test_results.json","final_predictions.csv",
  "final_run_manifest.json","country_results.csv","error_cases.csv",
  "efficiency_benchmark.csv","final_comparison_table.csv","final_comparison_table.json",
]:
    p = root/f
    print(f"  {'OK' if p.is_file() else 'MISSING'}: {p}")
for f in [
  "country_performance.png","false_positives.png","false_negatives.png",
  "missed_small_objects.png","localization_errors.png","confusion_matrix.png",
  "precision_recall_curves.png","accuracy_efficiency_tradeoff.png",
]:
    p = Path("figures/week6")/f
    print(f"  {'OK' if p.is_file() else 'MISSING'}: {p}")
print("  report:", Path("docs/week6_final_results_report.md").is_file())
PY

echo ""
echo "============================================================"
echo "Week 6 complete."
echo "Protocol: docs/final_evaluation_protocol.md"
echo "Report:   docs/week6_final_results_report.md"
echo "Results:  results/week6/"
echo "Figures:  figures/week6/"
echo "============================================================"
echo ""
echo "Reproduce:"
echo "  ./scripts/run_week6.sh --smoke"
echo "  ./scripts/run_week6.sh"
echo "  python scripts/run_final_evaluation.py"
echo "  python scripts/error_analysis.py"
echo "  python scripts/final_efficiency_benchmark.py"
echo "  python scripts/write_week6_tables.py"
