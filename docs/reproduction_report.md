# Reproduction report — primary final experiment

> **Degenerate check:** every compared metric is 0.0. The PASS results below only show that the evaluation code runs deterministically on this machine. They do **not** demonstrate that a meaningful result was reproduced.

_Generated: 2026-11-26T03:05:35.135437+00:00_

## Experiment

- Model: **YOLO11n @ 640 (Week 5 selected / control)** (`final_selected`)
- Checkpoint: `results/week5/yolo11n_640/seed_0/train/weights/best.pt`
- Protocol: `configs/final_evaluation.yaml` (`week6_final_v1`)
- Split evaluated: **Test** (`test`)
- imgsz=640, seed=0, conf_map=0.001, NMS IoU=0.7, operating conf=0.25
- Device: mps / Apple MPS
- Software: {'python': '3.11.1', 'platform': 'macOS-27.0.1-arm64-arm-64bit', 'torch': '2.15.0', 'cuda_available': False, 'ultralytics': '8.4.219'}

## Procedure

1. Load frozen protocol and locked confidence (validation-selected).
2. Confirm dataset split CSVs and hashes.
3. Load the selected checkpoint (no retrain).
4. Run Ultralytics `model.val` with the same imgsz / conf_for_map / IoU.
5. Compare to Week 6 `final_test_results.csv` row for `final_selected`.

## Split integrity

- Current hashes: `{'train': '3b76a2dc76c7d169', 'val': '7cbd768db64f4036', 'test': '3fd321b755c91c23'}`
- Week 5 recorded hashes: `{'train': '3b76a2dc76c7d169', 'val': '7cbd768db64f4036', 'test': '3fd321b755c91c23'}`
- Match: **True**

## Comparison table

| Metric | Original | Reproduced | Abs diff | Rel diff | Pass/Fail | Explanation |
|---|---|---|---|---|---|---|
| map50 | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| map50_95 | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| precision | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| recall | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| f1 | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| ap50_D00 | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| ap50_D10 | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| ap50_D20 | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |
| ap50_D40 | 0.0 | 0.0 | 0.0 | 0.0 | **PASS** | Within absolute tolerance |

**Failures:** 0 / 9

## Discrepancy notes

- Smoke-trained checkpoints yield near-zero mAP; tiny absolute differences across MPS/CUDA/CPU are expected.
- Nondeterminism (cuDNN, MPS, DataLoader workers) can change floating-point results slightly.
- This script does **not** retune thresholds or change the model after observing results.
- If checkpoint files are missing on another machine, copy `results/week*/**/weights/best.pt` or retrain under a **new** experiment ID — do not overwrite frozen Week 6 numbers.

## Pass criterion

- Absolute tolerance `0.0001` or relative tolerance `0.05` when |original| > 1e-6.
- Near-zero pairs (|o|<1e-5 and |r|<1e-5 with abs_diff≤1e-3) accepted with explanation.

CSV: `results/week7/reproduction_results.csv`
