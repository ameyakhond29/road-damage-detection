#!/usr/bin/env bash
# Week 3 end-to-end pipeline:
#   1) VOC → YOLO conversion
#   2) Dataset validation
#   3) Grouped 70/15/15 split + YAML
#   4) YOLOv8n baseline training
#   5) Results already exported by the train script
#
# Usage:
#   ./scripts/run_week3.sh
#   ./scripts/run_week3.sh --smoke          # short training for CI / no-GPU check
#   RAW_DIR=data/fixture/RDD2022 ./scripts/run_week3.sh --smoke
#
# Fails safely on the first error.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "${ALLOW_LEGACY_FIXTURE:-0}" != "1" ]]; then
  echo "LEGACY synthetic-fixture runner (Week 3). It does not use official RDD2022 data." >&2
  echo "Official pipeline: bash scripts/run_official_dataset_pipeline.sh" >&2
  echo "Set ALLOW_LEGACY_FIXTURE=1 to run it anyway (debugging only)." >&2
  exit 2
fi

# Prefer project venv if present
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"
SEED="${SEED:-6845}"
RAW_DIR="${RAW_DIR:-data/raw/RDD2022}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed/RDD2022_yolo}"
SMOKE=0
EXTRA_TRAIN_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Supported: --smoke" >&2
      exit 2
      ;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  EXTRA_TRAIN_ARGS+=(--smoke)
fi

# If official raw data is missing, fall back to fixture for pipeline verification.
if [[ ! -d "$RAW_DIR" ]] || [[ -z "$(ls -A "$RAW_DIR" 2>/dev/null || true)" ]]; then
  if [[ -d "data/fixture/RDD2022" ]]; then
    echo "WARNING: $RAW_DIR not found or empty."
    echo "Falling back to data/fixture/RDD2022 for pipeline verification only."
    echo "Do NOT report fixture metrics as official RDD2022 results."
    RAW_DIR="data/fixture/RDD2022"
  else
    echo "ERROR: RDD2022 not found at $RAW_DIR and no fixture available." >&2
    echo "Download from https://crddc2022.sekilab.global/data/ or Figshare DOI 10.6084/m9.figshare.21431547" >&2
    exit 1
  fi
fi

echo "============================================================"
echo "Week 3 pipeline"
echo "  ROOT=$ROOT"
echo "  RAW_DIR=$RAW_DIR"
echo "  PROCESSED_DIR=$PROCESSED_DIR"
echo "  SEED=$SEED"
echo "  SMOKE=$SMOKE"
echo "============================================================"

echo ""
echo "[1/5] Dataset conversion (VOC → YOLO) ..."
"$PYTHON" scripts/convert_voc_to_yolo.py \
  --raw-dir "$RAW_DIR" \
  --processed-dir "$PROCESSED_DIR" \
  --seed "$SEED"

echo ""
echo "[2/5] Dataset validation ..."
"$PYTHON" scripts/validate_dataset.py \
  --processed-dir "$PROCESSED_DIR" \
  --seed "$SEED"

echo ""
echo "[3/5] Grouped split creation (70/15/15, block=50, pHash) ..."
"$PYTHON" scripts/create_week3_splits.py \
  --processed-dir "$PROCESSED_DIR" \
  --splits-dir data/splits \
  --seed "$SEED"

echo ""
echo "[4/5] YOLOv8n baseline training ..."
"$PYTHON" scripts/train_yolov8_baseline.py \
  --data configs/rdd2022.yaml \
  --seed "$SEED" \
  "${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"}"

echo ""
echo "[5/5] Results export"
echo "  Conversion report: results/week3/conversion_report.json"
echo "  Validation:        results/week3/dataset_validation.json"
echo "  Split summary:     data/splits/split_summary.json"
echo "  Baseline run:      results/week3/yolov8n_baseline/"
echo "  Metrics:           results/week3/yolov8n_baseline/metrics.json"
echo ""
echo "Week 3 pipeline finished successfully."
