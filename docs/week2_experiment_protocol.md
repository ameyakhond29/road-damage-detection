# Week 2 — Experiment protocol

**Project:** Road Damage Detection Using Lightweight Object Detectors  
**Course:** DATA-255  
**Status:** Protocol and configs only — **do not fabricate results**; **do not run the full multi-seed matrix in Week 2**.

This document agrees with `docs/research_question.md` and `configs/experiment_manifest.csv`.

---

## 1. Final research question

Starting from a small detector, which single change provides the best accuracy per added computational cost on RDD2022: a **P2 stride-4 detection head**, a **lightweight neck-attention block**, or a **larger input size**? Do these improvements remain effective on countries not seen during training?

### Hypotheses

| ID | Statement |
|----|-----------|
| **H1** | The P2 detection head will improve AP50 for thin longitudinal and transverse cracks (D00, D10) more than the attention block at a similar computational cost. |
| **H2** | Improvements will be smaller on held-out countries than on in-distribution evaluation data, and at least one modification may lose its advantage. |

---

## 2. Dataset and split plan

| Item | Decision |
|------|----------|
| Dataset | RDD2022 |
| Classes | D00=0, D10=1, D20=2, D40=3 |
| Excluded | China_Drone; unsupported labels (D50, Repair, …) dropped at conversion |
| Negatives | Images with empty labels kept |
| Main splits | **Fixed** Week 1 files: `data/train_split.csv`, `data/val_split.csv`, `data/test_split.csv` |
| Split method | 70/15/15 by 50-filename blocks within country + phash duplicate merging; seed **6845** for data prep |
| YOLO lists | `data/processed/RDD2022_yolo/splits/{train,val,test}.txt` |
| Dataset YAML | `configs/rdd2022.yaml` |

### Country-held-out (generalization) plan

| Role | Countries |
|------|-----------|
| Train + val (selection) | Japan, India, Czech, China_MotorBike |
| Held-out evaluation | Norway, United_States (when labels/files exist) |

- Derive OOD CSVs with `python scripts/06_prepare_ood_splits.py` (**filters** fixed splits; does not re-split).  
- Paths: `data/ood/train_split.csv`, `data/ood/val_split.csv`, `data/ood/heldout_eval.csv`.  
- Dataset YAML stub: `configs/rdd2022_ood.yaml`.  
- **Checkpoint selection uses source-country val only.** Held-out and test are never used for selection.  
- Label all OOD results as **country-held-out**.

---

## 3. Experimental roles

### 3.1 Reference baseline

- **Model:** YOLOv8n, COCO-pretrained (`yolov8n.pt`)  
- **Input:** 640  
- **Recipe:** Standard Ultralytics detect training (`configs/baseline_yolov8n.yaml`)  
- **Purpose:** External reference so YOLO11 gains are interpretable  

### 3.2 Strong baseline

- **Model:** YOLO11n, COCO-pretrained (`yolo11n.pt`)  
- **Same** fixed split, epochs, augmentation, conf/NMS, and **validation-based** checkpoint selection  
- **Config:** `configs/baseline_yolo11n.yaml`  
- **Purpose:** Control for all team variants  

### 3.3 Team investigation

| Variant | Config | Factor |
|---------|--------|--------|
| YOLO11n + P2 stride-4 head | `configs/variant_p2.yaml` | Architecture |
| YOLO11n + lightweight neck attention | `configs/variant_attention.yaml` | Architecture |
| YOLO11n @ 800 px | `configs/variant_800px.yaml` | Resolution |
| YOLO11n + P2 + attention | `configs/variant_p2_attention.yaml` | Combined (`multi_factor: true`) |

Custom architecture YAMLs under `configs/models/` are **placeholders in Week 2**; wiring ships in Week 3. Do not silently train plain YOLO11n under a variant name.

---

## 4. Defined experiments

| Exp | Comparison | Manifest prefix | Answers |
|-----|------------|-----------------|---------|
| **1** | YOLOv8n vs YOLO11n @ 640 | `exp01_*` | Strong vs reference baseline |
| **2** | YOLO11n vs YOLO11n+P2 | `exp02_*` | H1 (P2 vs control) |
| **3** | YOLO11n vs YOLO11n+attention | `exp03_*` | H1 (attention vs control) |
| **4** | 640 vs 800 input (YOLO11n) | `exp04_*` | Resolution trade-off |
| **5** | P2 + attention combined | `exp05_*` | Interaction / upper bound |
| **OOD** | Train JP/IN/CZ/CN_M → eval NO/US | `ood_*` | H2 generalization |

**Main comparison (Week 3+):** Experiments 2–4 (and Exp 1 for context), multi-seed, accuracy **and** compute.

**Controlled ablations (minimum two):** Exp 2 (P2) and Exp 3 (attention); Exp 4 is the third controlled factor (resolution).

---

## 5. Held-constant conditions

Unless a row’s **intended factor** says otherwise, all runs share:

- Fixed split files and class mapping  
- Same data subset for a given track (main vs OOD)  
- Epochs = 100, patience = 30  
- Optimizer / LR schedule from Ultralytics defaults (`optimizer: auto`, `lr0=0.01`, `lrf=0.01`, …)  
- Batch **policy:** largest fitting batch for that `imgsz` (defaults: 16 @ 640, 8 @ 800); record actual batch  
- Same augmentation (see `configs/training_defaults.yaml`)  
- Same conf (0.001 for mAP) and NMS IoU (0.7)  
- Same checkpoint rule: **best validation mAP@0.5** (never test / held-out)  
- Same latency hardware and timing procedure within a comparison table  

Shared knobs: `configs/training_defaults.yaml`.  
Validator: `python scripts/check_experiment_config.py`.

---

## 6. Evaluation metrics

### Primary

- **mAP@0.5** averaged over D00, D10, D20, D40  

### Secondary

- Per-class AP50 (especially D00/D10 for H1)  
- mAP@0.5:0.95  
- Precision, Recall, F1  
- Small-box AP or small-damage recall (boxes below an area threshold defined at analysis time)  
- Parameters, GFLOPs, model size (MB)  
- GPU latency, CPU latency, throughput  
- Peak GPU memory  
- Wall-clock training time  

### Selection vs reporting

| Split | Use |
|-------|-----|
| Val | Checkpoint selection, hyperparameter freezes, early stopping |
| Test | **Final** in-distribution report only (after selection locked) |
| Held-out countries | **Final** OOD report only |

---

## 7. Compute trade-off plan

For every completed run, log:

| Quantity | How |
|----------|-----|
| Parameter count | Ultralytics model info / `model.info()` |
| GFLOPs | Ultralytics / thop at the run’s `imgsz` |
| Model size | `best.pt` file size on disk |
| GPU latency | Timed inference, batch=1, warm-up + N timed iters, fixed hardware |
| CPU latency | Same protocol on CPU |
| Throughput | Images/s at evaluation batch |
| GPU memory | Peak allocated during train and during timed inference |
| Training time | Wall clock from train start to end |

**Accuracy per cost** (analysis): ΔmAP50 / ΔGFLOPs and ΔmAP50 / ΔGPU_latency vs strong baseline. No numbers until runs exist.

---

## 8. Variability / reliability plan

| Rule | Detail |
|------|--------|
| Main comparison seeds | **0, 1, 2** |
| Data-prep / debug seed | **6845** (not for claiming main results) |
| Reporting | Mean ± std over seeds |
| Bootstrap | Paired bootstrap CIs, **1000** resamples, when implemented in Week 3/4 analysis code |
| One-seed runs | Clearly labeled **ONE-SEED** in notes/tables; no significance claims |
| Significance | Do **not** claim statistical significance unless the analysis supports it |

Manifest encodes 18 main multi-seed runs + 5 OOD runs (3× YOLO11n OOD multi-seed + 2× one-seed variant OOD) = **23 planned runs**. **Do not execute the full matrix in Week 2.**

---

## 9. Robustness plan

1. **Country-held-out** evaluation (H2).  
2. **Per-class** breakdown (crack vs pothole).  
3. **Small-damage** slice analysis.  
4. Optional qualitative failure gallery (false negatives on thin cracks).  

---

## 10. Risks and fallback plan

| Risk | Fallback |
|------|----------|
| Real RDD2022 not yet on disk | Keep protocol; run configs check + OOD CSV prep on fixture; **no published metrics** |
| Custom P2/attention YAML unfinished | Week 3 implements models; until then status stays `planned` |
| GPU OOM at 800 px | Lower batch under policy; record actual batch; do not change epochs/LR without logging a protocol deviation |
| Norway/US labels missing | Report OOD as unavailable; do not substitute test-set shopping |
| YOLO11n weights/API drift | Pin ultralytics version in the run log; fall back to YOLOv8n-only matrix with explicit scope change |
| Time budget | Priority order: Exp1 → Exp2 → Exp3 → Exp4 → Exp5 → OOD; one-seed pilots before multi-seed |
| Fixture confusion | Never cite fixture mAP as RDD2022 results |

---

## 11. Team responsibilities (suggested)

| Area | Owner (fill names) | Deliverable |
|------|--------------------|-------------|
| Data / splits / OOD prep | | Fixed splits intact; OOD CSVs; checker green |
| Baseline training (v8n, v11n) | | Exp 1 runs + compute logs |
| P2 head implementation | | `configs/models/yolo11n_p2.yaml` live + Exp 2 |
| Attention block | | `configs/models/yolo11n_attention.yaml` live + Exp 3 |
| Resolution + profiling | | Exp 4; latency/memory scripts |
| Analysis / plots / CIs | | Mean±std tables; no test leakage |
| Write-up | | Align report with this protocol |

---

## 12. Week 3 and Week 4 deliverables

### Week 3

- Implement P2 and attention model YAMLs (replace placeholders).  
- Training entry script that reads the manifest and refuses test-based selection.  
- One-seed **pilot** runs on real RDD2022 (labeled ONE-SEED).  
- Compute measurement utilities (GFLOPs, latency, memory).  
- OOD YOLO list files + first OOD pilot.  

### Week 4

- Full seeds 0/1/2 for Exps 1–5 (as hardware allows).  
- Country-held-out multi-seed for strong baseline (+ one-seed or multi-seed variants).  
- Mean±std tables, cost–accuracy plots, paired bootstrap if ready.  
- Draft results section; **test/held-out numbers only after selection freeze**.  

---

## 13. Manifest and configs

| File | Role |
|------|------|
| `configs/experiment_manifest.csv` | 23 planned runs |
| `configs/baseline_yolov8n.yaml` | Reference baseline |
| `configs/baseline_yolo11n.yaml` | Strong baseline |
| `configs/variant_p2.yaml` | Exp 2 |
| `configs/variant_attention.yaml` | Exp 3 |
| `configs/variant_800px.yaml` | Exp 4 |
| `configs/variant_p2_attention.yaml` | Exp 5 (`multi_factor`) |
| `configs/training_defaults.yaml` | Held-constant recipe |
| `scripts/check_experiment_config.py` | Protocol validator |

### Validate (Week 2 gate)

```bash
python scripts/06_prepare_ood_splits.py
python scripts/check_experiment_config.py
```

### Example future train command (Week 3 — not for full matrix yet)

```bash
# Illustrative only — implement train_from_manifest in Week 3
yolo detect train \
  cfg=configs/baseline_yolo11n.yaml \
  data=configs/rdd2022.yaml \
  seed=0 \
  project=results/experiments \
  name=exp01_yolo11n_640_s0
```

---

## 14. Explicit non-goals for Week 2

- Fabricating or estimating preliminary mAP/latency numbers  
- Using test or held-out metrics to choose a model  
- Running all 23 training jobs  
- Changing the fixed Week 1 split  
- Adding classes beyond D00–D40  
- Uploading RDD2022 to GitHub  
