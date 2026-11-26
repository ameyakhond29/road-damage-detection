#!/usr/bin/env bash
# Official RDD2022 data pipeline (no fixture data is used at any stage).
#
#   1. download / verify   scripts/download_rdd2022.py
#   2. audit               scripts/01_audit_dataset.py
#   3. convert VOC -> YOLO scripts/convert_voc_to_yolo.py
#   4. validate labels     scripts/validate_dataset.py
#   5. grouped splits      scripts/create_week3_splits.py
#   6. statistics          scripts/validate_dataset.py --require-splits (+ split/leakage summary)
#   7. real-data smoke     scripts/smoke_test_official_rdd2022.py   (skip with SKIP_SMOKE=1)
#
# Outputs: results/official_rdd2022/. Stops at the first failing command.
# Env: PYTHON (default python3), RDD2022_ROOT (raw root override), SKIP_DOWNLOAD=1,
#      SKIP_SMOKE=1, SMOKE_ARGS="--imgsz 640 --epochs 3", SEED (default 6845).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
SEED="${SEED:-6845}"
CONFIG="configs/rdd2022.yaml"
RAW="${RDD2022_ROOT:-data/raw/RDD2022}"
OUT="results/official_rdd2022"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"

trap 'echo "FAILED (exit $?): ${BASH_COMMAND}" | tee -a "$LOG" >&2' ERR

run() {
  echo "+ $*" | tee -a "$LOG"
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    echo "FAILED command: $*" | tee -a "$LOG" >&2
    exit 1
  fi
}

echo "=== Official RDD2022 pipeline $(date -u +%Y-%m-%dT%H:%M:%SZ) raw=$RAW ===" | tee "$LOG"

echo "=== [1/7] Download / verify ===" | tee -a "$LOG"
if [[ -n "${RDD2022_ROOT:-}" ]]; then
  echo "RDD2022_ROOT set; skipping download (using existing extraction)." | tee -a "$LOG"
elif [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  run "$PYTHON" scripts/download_rdd2022.py --skip-download
else
  run "$PYTHON" scripts/download_rdd2022.py
fi

echo "=== [2/7] Audit ===" | tee -a "$LOG"
run "$PYTHON" scripts/01_audit_dataset.py --data-root "$RAW" --config "$CONFIG"

echo "=== [3/7] Convert VOC -> YOLO ===" | tee -a "$LOG"
run "$PYTHON" scripts/convert_voc_to_yolo.py --data-root "$RAW" --config "$CONFIG"

echo "=== [4/7] Validate labels ===" | tee -a "$LOG"
run "$PYTHON" scripts/validate_dataset.py --config "$CONFIG"

echo "=== [5/7] Grouped splits (seed $SEED) ===" | tee -a "$LOG"
run "$PYTHON" scripts/create_week3_splits.py --config "$CONFIG" --seed "$SEED"

echo "=== [6/7] Statistics + leakage re-validation ===" | tee -a "$LOG"
run "$PYTHON" scripts/validate_dataset.py --config "$CONFIG" --require-splits --skip-image-check
run "$PYTHON" - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("data/splits/split_summary.json").read_text())
c = json.loads(Path("results/official_rdd2022/conversion_manifest.json").read_text())
stats = {"dataset": s["dataset"], "dataset_version": s["dataset_version"], "split_id": s["split_id"],
         "conversion": {k: c[k] for k in ("n_images_converted", "n_negatives_empty_labels", "n_objects",
                                          "class_counts", "per_country", "unsupported_labels_dropped")},
         "splits": s["splits"], "near_duplicate_purge": s["purged"],
         "n_purged_near_duplicates": s["n_purged_near_duplicates"], "leakage_passed": s["leakage_passed"]}
Path("results/official_rdd2022/dataset_statistics.json").write_text(json.dumps(stats, indent=2))
for k, v in s["splits"].items():
    print(f"{k}: images={v['n_images']} objects={v['n_objects']} negatives={v['n_negatives']} {v['class_counts']}")
print("Wrote results/official_rdd2022/dataset_statistics.json")
PY

echo "=== [7/7] Real-data smoke test ===" | tee -a "$LOG"
if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
  echo "SKIP_SMOKE=1 — smoke test skipped." | tee -a "$LOG"
else
  # shellcheck disable=SC2086
  run "$PYTHON" scripts/smoke_test_official_rdd2022.py --config "$CONFIG" ${SMOKE_ARGS:-}
fi

echo "=== Official RDD2022 pipeline finished OK — outputs in $OUT ===" | tee -a "$LOG"
