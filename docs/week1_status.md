# Week 1 status — Road Damage Detection

## Goal

Prepare and verify the complete **dataset pipeline** for RDD2022. Full detector experiments are deferred to later weeks; Week 1 only runs a tiny **smoke test**.

## Deliverables checklist

| Item | Status | Location |
|------|--------|----------|
| Project structure | Done | repo root |
| Dataset audit script | Done | `scripts/01_audit_dataset.py` |
| VOC → YOLO conversion | Done | `scripts/02_convert_annotations.py` |
| Filtering (no China_Drone, 4 classes, negatives kept) | Done | conversion defaults |
| Leakage-safe 70/15/15 splits | Done | `scripts/03_create_splits.py` |
| YOLO config | Done | `configs/rdd2022.yaml` (generated) |
| Smoke test (YOLOv8n) | Done | `scripts/05_smoke_test.py` |
| Documentation | Done | `docs/` |
| Reproducibility (CLI + seed 6845) | Done | all scripts |

## Verification run (this machine)

Official **RDD2022 was not present** under `data/raw/` at Week‑1 setup time. Pipeline execution was verified end-to-end on a **synthetic fixture** created by `scripts/00_create_fixture.py`.

**Do not cite fixture counts/mAP as RDD2022 results.** Re-run scripts 01–05 on the real download before reporting dataset statistics.

| Check | Result |
|-------|--------|
| Audit CSVs + figures | Written under `results/` and `figures/` |
| VOC→YOLO conversion | 361 images (China_Drone excluded); unsupported D50/Repair removed and counted |
| Splits | train/val/test CSVs + `duplicate_report.csv` |
| Smoke test | PASSED — train 3 epochs, val ran, `best.pt` saved, predictions montage written |
| Missing-data path | Prints download instructions (exit code 1) instead of a cryptic crash |

### Smoke-test command (used)

```bash
python scripts/05_smoke_test.py \
  --data configs/rdd2022.yaml \
  --epochs 3 \
  --imgsz 320 \
  --batch 4 \
  --max-train 300 \
  --max-val 100 \
  --seed 6845
```

Smoke-test metrics from the **fixture** run are stored in  
`results/smoke_test/smoke_test_summary.json` (near-zero mAP is expected on random synthetic boxes).

## Hardware and software (verification host)

| Item | Value |
|------|-------|
| Host | Apple M1 Pro (MPS); no NVIDIA CUDA on this Mac |
| Python | 3.11.1 |
| torch | 2.14.1 (`cuda=False`, `mps=True`) |
| ultralytics | 8.4.189 |
| Smoke device | `mps` (AMP off) |
| Default seed | 6845 |

Target GPUs for later experiments (same scripts):

| Target | Notes |
|--------|-------|
| NVIDIA RTX 4090 | CUDA + AMP auto-enabled |
| Google Colab T4 | CUDA + AMP |
| Kaggle T4 | CUDA + AMP |

Record Colab/Kaggle versions after install with:

```bash
python -c "import torch, ultralytics, sys; print(sys.version); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"
```

## Leakage-prevention method

1. **Group split:** blocks of 50 consecutive filenames within each country stay in one split.  
2. **Country stratification:** group assignment is done per country when enough groups exist.  
3. **Perceptual hash:** near-duplicates are unioned into the same group so they cannot cross train/val/test.  
4. Fixed seed **6845**.

## Problems encountered

1. **Real RDD2022 not downloaded yet** — used synthetic fixture for pipeline verification only.  
2. **Path with spaces** (`DATA 255`) — Ultralytics failed to resolve relative image list paths; fixed by writing **absolute** paths in `splits/*.txt`.  
3. **Ultralytics `path: ../...`** — relative parents were resolved incorrectly; `configs/rdd2022.yaml` uses an absolute dataset root, with a portable `path: .` copy at `data/processed/RDD2022_yolo/rdd2022.yaml`.  
4. **MPS deterministic warnings** — harmless; AMP remains off on MPS/CPU.

## Fallback plan

| Problem | Fallback |
|---------|----------|
| Cannot download full RDD2022 yet | Use `scripts/00_create_fixture.py` to verify code paths only; do not publish fixture metrics as RDD2022 results |
| Disk pressure | Convert with `--symlink-images` and/or process a subset via `--countries` |
| GPU OOM on smoke test | Lower `--batch` to 2 or keep `--imgsz 320` |
| Colab / Kaggle session limits | Run audit/conversion offline; only smoke-test in the notebook session |
| `imagehash` too slow on full set | Keep `--max-hamming 0` (exact phash buckets) |
| Ultralytics / torch install issues | Install CPU wheels first, then match CUDA build to the host driver |

## Exact commands to reproduce Week 1 (real RDD2022)

```bash
cd road-damage-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place RDD2022 under data/raw/RDD2022, then:
python scripts/01_audit_dataset.py --raw-dir data/raw/RDD2022 --seed 6845
python scripts/02_convert_annotations.py --raw-dir data/raw/RDD2022 --processed-dir data/processed/RDD2022_yolo --seed 6845
python scripts/03_create_splits.py --processed-dir data/processed/RDD2022_yolo --seed 6845
python scripts/04_write_yolo_config.py --processed-dir data/processed/RDD2022_yolo
python scripts/05_smoke_test.py --epochs 3 --imgsz 320 --batch 4 --seed 6845
```

### Fixture-only verification (no real dataset)

```bash
python scripts/00_create_fixture.py --seed 6845
python scripts/01_audit_dataset.py --raw-dir data/fixture/RDD2022 --seed 6845
python scripts/02_convert_annotations.py --raw-dir data/fixture/RDD2022 --processed-dir data/processed/RDD2022_yolo --seed 6845
python scripts/03_create_splits.py --processed-dir data/processed/RDD2022_yolo --seed 6845
python scripts/04_write_yolo_config.py --processed-dir data/processed/RDD2022_yolo
python scripts/05_smoke_test.py --epochs 3 --imgsz 320 --batch 4 --seed 6845
```

## Next weeks (out of scope for Week 1)

- Full YOLOv8n / YOLOv8s (or other lightweight detectors) training on the full filtered set  
- Country-wise and class-wise evaluation  
- Ablations (image size, augmentations, input resolution)
