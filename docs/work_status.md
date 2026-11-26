# Work status: completed work and current progress

**Project:** Road Damage Detection Using Lightweight Object Detectors (RDD2022)  
**Course:** DATA-255  
**Status date:** 2026-11-25  
**Companion documents:** `docs/progress_report.md` (full report), `docs/known_issues_and_resolutions.md` (issues log)

---

## 1. Summary

| Item | Status |
|---|---|
| Code and pipeline (data, training, evaluation, reproduction, packaging) | **Complete** and tested on the synthetic fixture |
| Documentation and reports | **Complete**, with provenance caveats |
| Results on the official RDD2022 dataset | **Not started.** The dataset is not in `data/raw/`. |
| Scientific conclusions (which model is best) | **Not supported yet.** All results come from fixture data and short smoke runs. |
| Open decisions made by the team | Test recount (amendment), git commit deferred |
| Open items needing real data, a GPU, or the team | See Section 5 |

**Bottom line:** the research pipeline is built, verified end to end on a synthetic fixture, and fully documented. The project cannot make claims about road-damage detection until the official RDD2022 data is run through it.

---

## 2. Completed work by area

### 2.1 Data pipeline (Weeks 1, 3)

| Work item | Result | Where |
|---|---|---|
| Dataset audit | Done (counts the raw fixture, including China_Drone) | `scripts/01_audit_dataset.py`, `results/audit_summary.json` |
| VOC → YOLO conversion | 361 images, 388 boxes, 120 negatives, D50/Repair dropped | `scripts/convert_voc_to_yolo.py`, `results/week3/conversion_report.json` |
| Leakage-safe split | Grouped 70/15/15, block size 50, seed 6845, pHash duplicates kept together | `scripts/create_week3_splits.py`, `data/splits/` |
| Split check | 53 near-duplicate pairs; 0 cross splits | `data/splits/split_summary.json` |
| Dataset validation | Passed | `results/week3/dataset_validation.json` |
| Modeling-set class figure | D00 93, D10 105, D20 103, D40 87 | `figures/class_distribution_modeling_set.png` |

### 2.2 Baselines (Weeks 3, 4)

| Work item | Result | Where |
|---|---|---|
| YOLOv8n reference baseline | Trained (3-epoch smoke, batch 4, seed 6845) | `results/week3/yolov8n_baseline/` |
| YOLO11n strong baseline | Trained (3-epoch smoke) | `results/week4/yolo11n_baseline/` |
| Baseline comparison | Saved table and figures | `results/week4/baseline_comparison.csv`, `figures/week4/` |

### 2.3 Architecture changes and ablations (Week 5)

| Work item | Result | Where |
|---|---|---|
| P2 head model YAML | Implemented; Detect layers 4 scales | `configs/models/yolo11n-p2.yaml` |
| Attention model YAML | Implemented (C2PSA on P3) | `configs/models/yolo11n-attn.yaml` |
| Combined model YAML | Implemented; attention moved from P2 to P3 after an out-of-memory error at 800 px | `configs/models/yolo11n-p2-attn.yaml` |
| Unit tests | **8 of 8 pass** (in a clean environment too) | `tests/test_week5_models.py` |
| Five ablation runs | Seed 0 only, 1-epoch smoke each. Seeds 1 and 2 not run. | `results/week5/` |
| Efficiency measurements | Parameters, GFLOPs, CPU latency for all five variants | `results/week5/efficiency_results.csv` |
| Selection decision | YOLO11n @ 640 (provisional, validation only) | `results/week5/model_selection_decision.md` |

### 2.4 Final evaluation (Week 6)

| Work item | Result | Where |
|---|---|---|
| Frozen protocol | Written and versioned (`week6_final_v1`) | `configs/final_evaluation.yaml`, `docs/final_evaluation_protocol.md` |
| Confidence threshold lock | 0.25, chosen on validation. All six candidates had F1 = 0, so this was a tie-break. | `results/week6/CONFIDENCE_THRESHOLD_LOCKED.json` |
| Final test evaluation | Run in smoke mode. Metrics cover the full split. | `results/week6/final_test_results.csv` (original, kept) |
| Protocol amendment `week6_full_test_counts_v1` | FP/FN, error analysis, and country rows re-computed on the full 50-image test split. Model, thresholds, and selection unchanged. | `results/week6/amendment_full_test_counts/` |
| Error analysis | Re-run on the full split: 274 errors across 50 images | `results/week6/error_cases.csv`, `results/week6/error_analysis_summary.json` (smoke outputs archived in `results/week6/archive_smoke_12img/`) |
| Efficiency benchmark | On Apple MPS (not T4). Latency 21.8 ms mean. | `results/week6/efficiency_benchmark.csv` |
| Final comparison tables | Validation and Test in separate columns | `results/week6/final_comparison_table.csv` |
| Final report | Generated with provenance and amendment sections | `docs/week6_final_results_report.md` |

### 2.5 Reproducibility, environment, and packaging (Week 7)

| Work item | Result | Where |
|---|---|---|
| Environment setup | `requirements.txt`, `environment.yml`, `scripts/setup_environment.sh`, `scripts/check_environment.py` | repository root, `scripts/` |
| Pinned environment | `requirements-lock.txt` from the verified clean environment (torch 2.15.0, ultralytics 8.4.219) | repository root |
| Clean-environment check | Fresh venv from `requirements.txt`, 8 of 8 tests pass, reproduction runs | verified; see `docs/known_issues_and_resolutions.md` C4 |
| Reproduction check | Runs. **Degenerate:** every compared value is 0.0, so this shows determinism, not reproduction. Flagged in code and report. | `results/week7/reproduction_results.csv`, `docs/reproduction_report.md` |
| Dataset card | Written, with a provenance warning | `docs/dataset_card.md` |
| Experiment log and manifest | Written | `docs/experiment_log.md`, `results/week7/experiment_manifest.csv` |
| Final results package | Tables, figures, and results copied with a manifest | `results/final/`, `figures/final/`, `tables/final/` |
| Repository audit | Written | `docs/repository_audit.md` |
| Quality checklist | Rewritten with actual statuses (no unverified "Pass" entries) | `docs/week7_quality_checklist.md` |
| AI disclosure | Written with the required text | `docs/AI_use.md` |
| Team contributions | Drafted; **not yet verified by team members** | `docs/team_contributions.md` |
| Presentation outline | Written (12 slides). Do not use current result figures as evidence. | `presentation/week7_slide_outline.md` |
| README | Rewritten, with a status and provenance section | `README.md` |
| Colab / Kaggle guidance | Written in the README and setup docs | `README.md`, `scripts/setup_environment.sh --colab` |

### 2.6 Quality review and corrections (this session)

| Work item | Result |
|---|---|
| Figure review | All 39 PNG figures inspected. Most result plots are empty or flat, and the galleries show fixture noise. Documented. |
| Stale claims corrected | "Fixed" dataset path claim, "Pass" checklist entries, "limited RDD2022 subsets" wording, and "AI drafts reviewed" claim |
| Test coverage | Discovered that the final test run was in smoke mode. Resolved by amendment (see Section 2.4). |
| Issues log | `docs/known_issues_and_resolutions.md`: each issue with its evidence, status, and remaining action |

---

## 3. Progress by week

| Week | Goal | Progress | What is still missing |
|---|---|---|---|
| 1 | Dataset audit, conversion, split, smoke test, docs | Pipeline complete on the fixture | Real RDD2022 run |
| 2 | Protocol, configs, manifest, config checker | Complete | None for the protocol |
| 3 | Preprocessing, validation, YOLOv8n baseline | Pipeline complete on the fixture | Real data; full-length training |
| 4 | YOLO11n baseline, fair comparison | Pipeline complete on the fixture | Real data; full-length training |
| 5 | P2, attention, 800 px, combined ablations | Models implemented and tested. Five smoke runs (seed 0, 1 epoch). | Full training, seeds 1 and 2, real data |
| 6 | Frozen evaluation, error analysis, efficiency, tables | Protocol frozen. Test metrics and full-split counts produced under amendment. Efficiency on MPS. | Real data; T4 efficiency; re-evaluation after real training |
| 7 | Reproducibility, docs, packaging, presentation | Scripts, docs, clean-environment check, packaging, and figure review complete | Colab run; team verification; real-data reproduction; git commit |

---

## 4. Key results (fixture only; not evidence about road damage)

| Result | Value | Source |
|---|---|---|
| Final model (YOLO11n @ 640) validation mAP@0.5 | 8.1e-06 | `results/week6/final_comparison_table.csv` |
| Final model test mAP@0.5 (full split) | 0.0 | `results/week6/final_test_results.csv` |
| Final model test counts, full split | 55 GT, 0 TP, 219 FP, 42 FN, 13 localization-poor | `results/week6/amendment_full_test_counts/full_test_counts.json` |
| Ablations better than control on validation | None | `results/week5/seed_summary.csv` |
| Efficiency (Apple MPS) | 2.6M parameters, 6.57 GFLOPs, 21.8 ms | `results/week6/efficiency_benchmark.csv` |

---

## 5. Remaining work

### 5.1 Needs real RDD2022 data or a GPU

1. Place the official RDD2022 download under `data/raw/RDD2022/`, then re-run conversion, validation, and split checks.
2. Train the baselines and the five Week 5 variants at full budget (100 epochs, early stopping, AMP on CUDA), with seeds 0, 1, and 2.
3. Re-select on validation if the ranking changes, and freeze a new protocol version.
4. Run one full test evaluation for the new frozen models.
5. Benchmark efficiency on a Colab or Kaggle T4.
6. Run the Colab or Kaggle setup end to end.

### 5.2 Needs the team

1. Each member confirms or corrects their section of `docs/team_contributions.md`.
2. Decide when to make the first git commit. The team chose not to commit yet.
3. Recreate the workspace `.venv`, which currently has no torch installed. Or use `requirements-lock.txt`.

### 5.3 Rules for any further test reading

- The test split has now been read twice: once in the original smoke run, and once for the full-split amendment.
- Any further test reading needs a new documented amendment or a new protocol version.
- Do not re-run the Week 6 test evaluation to change the model or thresholds.

---

## 6. Deliverables checklist

| Deliverable | Status |
|---|---|
| Data pipeline and splits | Done (fixture) |
| Baselines | Done (fixture, smoke) |
| Week 5 architectures, tests, ablations | Done for seed 0 (smoke). Seeds 1 and 2 pending. |
| Week 6 frozen evaluation and amendment | Done (fixture) |
| Week 6 final report | Done (with caveats) |
| Week 7 documentation and packaging | Done |
| Reproducibility | Environment verified; reproduction degenerate |
| Presentation outline | Done (figures not yet replaced with real results) |
| Real RDD2022 results | Not started |
| T4 efficiency | Not started |
| Colab run | Not started |
| Team verification | Not started |
