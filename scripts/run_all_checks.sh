#!/usr/bin/env bash
# Week 7 quality / reproducibility checks
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
PYTHON="${PYTHON:-python3}"

echo "=== [1] Environment ==="
"$PYTHON" scripts/check_environment.py

echo "=== [2] Unit tests ==="
"$PYTHON" -m unittest discover -s tests -v

echo "=== [3] Protocol / split sanity ==="
"$PYTHON" - <<'PY'
from pathlib import Path
import yaml, hashlib, json
root = Path('.')
# Official dataset config: tracked, portable, no machine paths.
cfg = yaml.safe_load((root/'configs/rdd2022.yaml').read_text())
assert cfg['dataset'] == 'official_rdd2022' and not str(cfg['raw_root']).startswith('/'), cfg
# Legacy (synthetic fixture) Week 6 frozen protocol.
proto = yaml.safe_load((root/'configs/final_evaluation.yaml').read_text())
assert proto.get('protocol_frozen') is True
assert proto['final_model'].get('test_used_for_selection') is False
assert Path(proto['final_model']['checkpoint']).is_file()
from src.paths import ensure_legacy_fixture_data_yaml, legacy_fixture_processed_dir, legacy_fixture_splits_dir
data = yaml.safe_load(ensure_legacy_fixture_data_yaml().read_text())
assert Path(data['path']).resolve() == legacy_fixture_processed_dir().resolve(), data.get('path')
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
print('legacy fixture split hashes', {s: h(legacy_fixture_splits_dir()/f'{s}.csv') for s in ('train','val','test')})
summary = root/'data/splits/split_summary.json'
if summary.is_file() and json.loads(summary.read_text()).get('dataset') == 'official_rdd2022':
    print('official split hashes', {s: h(f'data/splits/{s}.csv') for s in ('train','val','test')})
lock = root/'results/week6/CONFIDENCE_THRESHOLD_LOCKED.json'
assert lock.is_file()
print('conf lock (legacy fixture)', json.loads(lock.read_text())['selected_conf'])
print('PASS protocol/split checks')
PY

echo "=== [4] Package final artifacts ==="
"$PYTHON" scripts/package_final_results.py

echo "=== [5] Reproduce main result ==="
"$PYTHON" scripts/reproduce_main_result.py

echo "=== [6] Write Week 7 docs helpers if present ==="
"$PYTHON" scripts/write_week7_docs.py || true

echo "ALL CHECKS FINISHED"
