# Project Progress Report — Road Damage Detection Using Lightweight Object Detectors

**Course:** DATA-255 (Deep Learning / Computer Vision)  
**Dataset:** RDD2022 (classes D00, D10, D20, D40)  
**Report date:** 2026-11-25  
**Repository:** `road-damage-detection/`  
**Team (as drafted in `docs/team_contributions.md`, not yet verified by each member):** Ameya (data pipeline, conversion, splits, validation) · Prathmesh (training pipeline, baselines) · Tejas (architecture changes, ablations) · Vrishin (evaluation, benchmarking, error analysis)

---

## 0. Data caveats (read first)

**1. All experimental results in this repository were produced on a synthetic fixture, not on the official RDD2022 dataset.**

- The official RDD2022 download has never been placed in `data/raw/`, which is empty.
- The processed dataset (361 images, 388 boxes, 120 negatives) was generated from `data/fixture/RDD2022/` by `scripts/00_create_fixture.py`. The `source_image` column of `manifest.csv` points to the fixture.
- Every mAP value is near zero. Checkpoints come from 1–3 epoch smoke training. These values are not evidence about road-damage detection.

**2. The original final test run used smoke mode. The counts were then re-computed on the full test split under a documented amendment.**

- `results/week6/final_run_manifest.json` records `"smoke": true`. The original FP/FN counts (72 FP, 11 FN, 0 TP from 73 detections) and the error analysis covered only the **first 12 test images**. The per-country rows were capped at 8 images per country.
- The mAP, precision, recall, F1, per-class AP, and small-object AP were already computed on the full 50-image test split.
- **Amendment `week6_full_test_counts_v1`** (decided by the team): FP/FN counts, error analysis, and country rows were re-computed on all 50 test images with the **same** checkpoint, validation-locked threshold (0.25), NMS IoU 0.7, and matching IoU 0.5. Model, thresholds, and selection were not changed, and nothing was retrained. Record: `results/week6/amendment_full_test_counts/full_test_counts.json`.
- Full-split results: 55 ground-truth boxes, 219 FP, 42 FN, 13 localization-poor, 0 TP. The error analysis reports 274 errors across 50 images, which reconciles (219 + 42 + 13).
- The test split was read a second time for these counts after the original results were observed. Any write-up must say so.

The issues log (`docs/known_issues_and_resolutions.md`) lists every known problem, its evidence, its status, and what remains.

---

## 1. Project objective and research question

**Objective:** Build a reproducible pipeline for lightweight road-damage detection on RDD2022, then measure which controlled change to a small YOLO detector improves accuracy per added compute, especially for small cracks and potholes.

**Research question:** Starting from a small detector, which single change provides the best accuracy per added computational cost: a P2 (stride-4) detection head, a lightweight neck-attention block, or a larger input size (800 px)? Does a combined model justify its cost?

**Protocol commitments:**
- Model and checkpoint selection use validation data only.
- The test set is evaluated once under a frozen protocol (see the caveat in Section 0.2).
- Seeds 0, 1, 2 for principal comparisons (only seed 0 has been run).
- Grouped, leakage-safe 70/15/15 split (seed 6845).

---

## 2. Status by week

| Week | Focus | Status | Notes |
|---|---|---|---|
| 1 | Dataset audit, VOC→YOLO conversion, filtering, leakage-safe split, smoke test, docs | **Complete (pipeline)** | Verified on fixture. Real RDD2022 not run. |
| 2 | Experiment protocol, configs, manifest, config checker | **Complete** | Model YAMLs for P2/attention were placeholders at this stage. |
| 3 | Dataset finalization, validation, split materialization, YOLOv8n baseline | **Complete (pipeline)** | Fixture data; 3-epoch smoke training. |
| 4 | YOLO11n strong baseline, fair comparison, evaluation (val + test) | **Complete (pipeline)** | Fixture data; 3-epoch smoke training. |
| 5 | P2 / attention / 800 px / combined ablations, efficiency, decision file, unit tests | **Partially complete** | Seed 0 only, 1-epoch smoke run per variant. Seeds 1 and 2 not run. |
| 6 | Freeze protocol, one-shot test evaluation, country analysis, error analysis, efficiency benchmark, final tables | **Pipeline complete; counts amended to full split** | Test mAP from the full split. FP/FN counts, error analysis, and country rows re-computed on the full split under amendment `week6_full_test_counts_v1`. Efficiency measured on Apple MPS, not T4. |
| 7 | Reproducibility, repository audit, environment scripts, dataset card, experiment log, packaging, presentation outline, docs | **Pipeline complete; checks now verified where possible** | Clean-environment install and unit tests verified. Figures reviewed. Reproduction check is degenerate. See Section 9. |

---

## 3. Work completed, by area

### 3.1 Data pipeline (Weeks 1, 3)

- Audit scripts and dataset statistics (`scripts/01_audit_dataset.py`, `results/audit_summary.json`, `results/class_distribution.csv`, `results/box_statistics.csv`). The audit counts the raw fixture XMLs including China_Drone, so its class counts (108/123/110/102) do **not** describe the modeling set.
- VOC → YOLO conversion (`scripts/convert_voc_to_yolo.py`, `src/dataset/convert.py`). Excludes China_Drone, keeps D00/D10/D20/D40, drops unsupported labels (D50, Repair) with counts, keeps negative images as empty label files.
- Leakage-safe grouped split (`scripts/create_week3_splits.py`, `src/dataset/splits.py`): blocks of 50 consecutive filenames per country, pHash near-duplicates forced into the same split, seed 6845.
- Dataset validation (`scripts/validate_dataset.py`): passed. 361 images, 0 missing labels, 0 bad lines, 0 out-of-bounds boxes.

Modeling-set class distribution (`results/week3/dataset_validation.json`; figure `figures/class_distribution_modeling_set.png`):

| Class | Boxes |
|---|---|
| D00 | 93 |
| D10 | 105 |
| D20 | 103 |
| D40 | 87 |
| **Total** | **388** |

Split summary (`data/splits/split_summary.json`):

| Split | Images | Boxes | Negatives | Countries |
|---|---|---|---|---|
| Train | 252 | 275 | 81 | China_MotorBike 50, Czech 60, India 60, Japan 61, Norway 11, United_States 10 |
| Val | 59 | 58 | 23 | China_MotorBike 10, Norway 49 |
| Test | 50 | 55 | 16 | United_States 50 |

Duplicate leakage check: 53 near-duplicate pairs; 0 pairs cross splits (`passed: true`).

### 3.2 Baselines (Weeks 3, 4)

- **YOLOv8n reference baseline** (`scripts/train_yolov8_baseline.py`, `results/week3/yolov8n_baseline/`).
- **YOLO11n strong baseline** (`scripts/train_yolo11_baseline.py`, `results/week4/yolo11n_baseline/`).
- Comparison and evaluation tooling (`scripts/compare_baselines.py`, `scripts/evaluate_week4.py`, `results/week4/baseline_comparison.csv`).

Both were run for 3 epochs, batch 4, seed 6845, on Apple MPS, against the fixture. They are smoke runs.

### 3.3 Model architecture changes (Week 5)

Implemented as Ultralytics-loadable YAMLs (replacing the Week 2 placeholders):

| File | Architecture |
|---|---|
| `configs/models/yolo11n-p2.yaml` | YOLO11n with a P2 (stride-4) Detect head |
| `configs/models/yolo11n-attn.yaml` | YOLO11n with C2PSA attention on the P3 neck feature |
| `configs/models/yolo11n-p2-attn.yaml` | P2 head + C2PSA on P3 (combined, 800 px) |

Design decision: the combined model's C2PSA was first placed on the P2 feature map. At 800 px on Apple MPS this ran out of memory (about 12 GiB attention buffer). Attention was moved to P3, matching the single-factor attention ablation. This is documented in the config and in `src/models/resolve.py`.

Unit tests (`tests/test_week5_models.py`): **8 of 8 pass** (verified again in a clean environment, Section 3.6).

### 3.4 Week 5 ablation experiments

Five configs (`configs/week5_*.yaml`) share the same split and class mapping. Every run was a 1-epoch smoke run on Apple MPS.

| Experiment | Input | Batch | Epochs | Seeds completed | Status (from `metrics.json`) |
|---|---|---|---|---|---|
| YOLO11n @ 640 (control) | 640 | 4 | 1 | 1 (seed 0) | completed_smoke |
| YOLO11n + P2 @ 640 | 640 | 4 | 1 | 1 (seed 0) | completed_smoke |
| YOLO11n + attention @ 640 | 640 | 4 | 1 | 1 (seed 0) | completed_smoke |
| YOLO11n @ 800 | 800 | 2 | 1 | 1 (seed 0) | completed_smoke |
| Combined (P2 + attention @ 800) | 800 | 1 | 1 | 1 (seed 0) | completed_smoke |

Seeds 1 and 2 were **not run**. The seed summary reports `n_seeds_completed = 1` for every variant.

Validation mAP@0.5 (seed 0, smoke, `results/week5/seed_summary.csv`):

| Experiment | Validation mAP@0.5 |
|---|---|
| YOLO11n @ 640 | 8.1e-06 |
| + P2 @ 640 | 0.0 |
| + attention @ 640 | 0.0 |
| @ 800 | 0.0 |
| Combined @ 800 | 0.0 |

Efficiency (`results/week5/efficiency_results.csv`, CPU, seed-0 checkpoints):

| Experiment | Parameters | GFLOPs | Latency (ms, CPU, mean) |
|---|---|---|---|
| YOLO11n @ 640 | 2,605,612 | 6.57 | 83.3 |
| + P2 @ 640 | 2,667,216 | 10.46 | 104.7 |
| + attention @ 640 | 2,621,964 | 10.72 | 129.3 |
| @ 800 | 2,605,612 | 6.57 | 109.6 |
| Combined @ 800 | 2,683,568 | 14.61 | 293.9 |

Decision (`results/week5/model_selection_decision.md`): YOLO11n @ 640 was selected as the provisional candidate because it was the only variant with a non-zero validation mAP, and it ties with the others on the tie-break rule. Selection used validation metrics only. The decision file states that all runs are smoke runs.

### 3.5 Frozen final evaluation (Week 6)

- Frozen protocol: `configs/final_evaluation.yaml` and `docs/final_evaluation_protocol.md`. Final model YOLO11n @ 640, NMS IoU 0.7, conf-for-mAP 0.001, operating confidence 0.25.
- Confidence lock (`results/week6/CONFIDENCE_THRESHOLD_LOCKED.json`): swept on validation over six thresholds (0.05, 0.10, 0.25, 0.40, 0.50, 0.65). Criterion `max_f1_then_precision`. **Every threshold gave F1 = 0**, so 0.25 was chosen by the tie-break, not by an informative sweep.
- Test evaluation (`scripts/run_final_evaluation.py`) was run with `--smoke`. Recorded in `results/week6/TEST_EVALUATION_COMPLETED.json` and `results/week6/final_run_manifest.json`. Counts were then re-computed on the full split under amendment `week6_full_test_counts_v1` (Section 0.2).

Final model, test split (`results/week6/final_test_results.csv` for the metrics, and `results/week6/amendment_full_test_counts/` for full-split counts):

| Metric | Value | Coverage |
|---|---|---|
| mAP@0.5 | 0.0 | Full 50-image split (Ultralytics `model.val`) |
| mAP@0.5:0.95 | 0.0 | Full 50-image split |
| Precision | 0.0 | Full 50-image split |
| Recall | 0.0 | Full 50-image split |
| F1 | 0.0 | Full 50-image split |
| Detections | 232 | Full 50-image split (amendment) |
| False positives | 219 | Full 50-image split (amendment) |
| False negatives | 42 | Full 50-image split (amendment) |
| Localization-poor matches | 13 | Full 50-image split (amendment) |
| True positives | 0 | Full 50-image split (amendment) |
| Ground-truth boxes | 55 | Full 50-image split |

The original smoke-run counts (73 detections, 72 FP, 11 FN) covered only the first 12 images and are kept for transparency.

Validation values for the same model: mAP@0.5 = 8.1e-06, precision = 1.04e-04, recall = 0.038.

Baselines, same frozen protocol (`results/week6/final_comparison_table.csv`):

| Model | Val mAP@0.5 | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---|---|---|
| YOLOv8n (Week 3) | 8.73e-05 | 8.35e-06 | 2.14e-06 |
| YOLO11n (Week 4) | 0.0 | 1.13e-06 | 6.78e-07 |
| Final selected (YOLO11n @ 640) | 8.10e-06 | 0.0 | 0.0 |

Country analysis: the test split is entirely United_States under the fixed split. The full-split country row (`results/week6/amendment_full_test_counts/full_test_country_results.csv`) covers 50 images and 55 objects, with box recall 0.0. Official per-country mAP is Pending. No new split was created. The original smoke country rows (8-image cap) are in `results/week6/country_results.csv`. The country performance figure still reflects the smoke-run data and is empty.

Efficiency benchmark (`results/week6/efficiency_benchmark.csv`): measured on **Apple MPS**, not the NVIDIA T4 the protocol requires. The file labels this.

### 3.6 Reproducibility, environment, and packaging (Week 7)

- Environment: `requirements.txt`, `environment.yml`, `scripts/setup_environment.sh`, `scripts/check_environment.py`.
- **New:** `requirements-lock.txt` pins the versions used in the clean-environment check (torch 2.15.0, ultralytics 8.4.219, numpy 1.26.4, pandas 3.0.6). Earlier results were produced with torch 2.14.1 and ultralytics 8.4.189 (per `results/week6/final_run_manifest.json`).
- **Clean-environment check (new):** a fresh virtual environment was created from `requirements.txt` on macOS arm64. `python -m unittest discover -s tests -v` passed **8 of 8**. The reproduction script ran in that environment.
- Data config: Ultralytics ignores repo-relative `path` values in dataset YAMLs. `src/paths.py::ensure_rdd2022_data_yaml()` writes an absolute path for each machine. `configs/rdd2022.yaml` is now gitignored because it is machine-generated.
- Reproduction (`scripts/reproduce_main_result.py`): reloads the frozen YOLO11n @ 640 checkpoint, re-runs test evaluation, and compares with the saved Week 6 values. Output in `results/week7/reproduction_results.csv`, `results/week7/reproduction_results.json`, and `docs/reproduction_report.md`.
- **Reproduction check is degenerate.** Every compared value is 0.0, so the 9 of 9 PASS results show only that the evaluation runs deterministically on this machine. The script now flags this (`degenerate_all_zero: true`) and the report carries a warning. A meaningful reproduction needs real data.
- Packaging (`scripts/package_final_results.py`): copies tables, JSON, and figures into `results/final/`, `figures/final/`, and `tables/final/`, with a manifest. The class-distribution figure in the package is now the modeling-set figure.
- Documentation: `docs/repository_audit.md`, `docs/dataset_card.md`, `docs/experiment_log.md`, `results/week7/experiment_manifest.csv`, `docs/final_results_summary.md`, `docs/AI_use.md`, `docs/team_contributions.md`, `docs/week7_quality_checklist.md`, `docs/known_issues_and_resolutions.md`, `presentation/week7_slide_outline.md`, and the updated `README.md` and `data/README.md`.

### 3.7 Figure review (new)

All 39 PNG figures in `figures/` were inspected in contact sheets. Findings are in `docs/known_issues_and_resolutions.md`, Section D. In summary:

- **Empty or flat:** precision-recall curves (AP 0.000), confidence sweeps (flat at 0), country performance bars, Week 5 training curves (1 epoch), TP-only confusion matrix.
- **Single non-zero bar:** Week 5 ablation comparisons (only the control is non-zero, at ~8e-6).
- **Not road scenes:** the qualitative galleries (false positives, false negatives, missed small objects, localization errors, smoke predictions) show coloured noise from the synthetic fixture.
- **Corrected:** the class-distribution figure in the package now shows the modeling set (388 boxes), not the raw audit count (443 boxes).
- Titles and axis labels are present on all figures.

**Recommendation:** do not use the current figures in the presentation as evidence of model performance.

---

## 4. Summary of verified results

All values below come from saved files and are fixture-scale.

| Question | Answer from saved results |
|---|---|
| Which model was selected? | YOLO11n @ 640 (provisional; selected on validation, where all values are near zero) |
| Did any ablation beat the control on validation mAP? | No (all ablations 0.0 vs control 8.1e-06; seed 0 only) |
| Did the final model achieve a non-zero test mAP@0.5? | No (0.0) |
| Did test evaluation use test labels for selection? | No (`test_used_for_selection: false`; confidence lock uses validation only) |
| Was the final test evaluation complete? | Metrics: full split. FP/FN counts, error analysis, and country rows: full split after amendment `week6_full_test_counts_v1`. The original smoke run is kept for transparency. |
| Are efficiency numbers on the T4 GPU? | No (Apple MPS for Week 6, CPU for Week 5) |
| Are seeds 1 and 2 complete? | No (not run) |

---

## 5. Problems encountered and how they were handled

| Problem | Resolution |
|---|---|
| Official RDD2022 not available on the development machine | Pipeline verified on a synthetic fixture; results labeled as fixture/smoke. **Still open for real results.** |
| Ultralytics ignored relative dataset path | Absolute path generated per machine via `ensure_rdd2022_data_yaml()`. Config is gitignored. |
| Combined P2 + attention at 800 px ran out of memory on MPS | Attention moved from P2 to P3 (documented in config and code). |
| MPS evaluation of one checkpoint ran out of memory | Evaluation retried on CPU for that checkpoint (`evaluate_week5_ablation.py`). |
| Week 5 efficiency script stalled on MPS | Re-run on CPU with reduced iterations. |
| Confidence threshold sweep was uninformative (all F1 = 0) | Recorded as-is; tie-break applied; disclosed. |
| Final test run was launched in smoke mode | Discovered during this audit. Team chose a full-split recount recorded as amendment `week6_full_test_counts_v1`. Model, thresholds, and selection unchanged. Original outputs kept. |
| Country table showed 8 images for a 50-image split | Traced to the smoke cap. Full-split country rows produced by the amendment. |
| Week 7 checklist and docs claimed "Pass" for items not verified | Checklist rewritten with actual status. Claims corrected. |
| Reproduction PASS was trivial (all zeros) | Degenerate case now detected and flagged in code and report. |
| Class-distribution figure described the raw audit, not the modeling set | New modeling-set figure created; packaging updated. |
| Dependencies unpinned; earlier results used different versions | `requirements-lock.txt` added from the verified clean environment. |
| Sandbox blocked torch (segfault on import) during audit | Re-ran the clean-environment checks outside the sandbox. Not a project issue. |

---

## 6. Known limitations and threats to validity

1. **Data:** No real RDD2022 images were used. This is the most important limitation. All metrics are on a synthetic fixture.
2. **Test reading:** The test split was read twice: once in the original smoke-mode run and once more for the full-split counts (amendment). The second reading came after the original results were observed. Model, thresholds, and selection did not change.
3. **Training budget:** 1–3 epochs for smoke runs, against a planned 100 epochs with early stopping.
4. **Seeds:** Only seed 0 for Week 5 ablations; seeds 1 and 2 are missing. Week 3/4 baselines used seed 6845.
5. **Test split:** Under the fixed split the test set has one country (United_States). No country-held-out generalization claim can be made.
6. **Efficiency:** Not measured on the NVIDIA T4 required by the protocol. MPS and CPU numbers are not comparable to T4 numbers.
7. **Small-object AP:** Defined here as AP on objects with normalized area < 0.02. This is project-specific and not directly comparable to other papers.
8. **Reproduction:** Only the evaluation step is reproduced (checkpoint reloaded, no retraining). The result is degenerate (all zeros).
9. **Software versions:** Earlier results used torch 2.14.1 and ultralytics 8.4.189. The clean-environment check used torch 2.15.0 and ultralytics 8.4.219. Numerics may differ slightly.
10. **Colab / Kaggle:** Setup instructions were written, but no Colab or Kaggle run has been performed.
11. **Figures:** Most result figures are empty or uninformative at this scale (Section 3.7).

---

## 7. Remaining work

Priority order:

1. **Obtain real RDD2022** and place it in `data/raw/RDD2022/` (or Google Drive on Colab). Re-run conversion, validation, and split verification.
3. **Train on real data** with the full protocol (100 epochs, early stopping, AMP on CUDA): YOLOv8n, YOLO11n, and the five Week 5 variants, with seeds 0, 1, 2 for the principal comparison.
4. **Re-select** the checkpoint and threshold on validation if the ranking changes, and freeze a new protocol version. Do not reuse the current test results as final.
5. **Run the one-shot test evaluation** on the new frozen models with full coverage.
6. **Benchmark efficiency on a T4** (Colab or Kaggle).
7. **Run the Colab/Kaggle setup end to end** and record any issues.
8. **Confirm team contributions** with each member (Section 8).
9. **Make an initial git commit** if the team agrees (no commits exist yet).
10. **Recreate the workspace `.venv`** (it currently has no torch installed) or use `requirements-lock.txt`.

---

## 8. Team contributions

`docs/team_contributions.md` was drafted with AI assistance. **The attributions below are the draft's wording and have not been verified by the team.** The draft's statement that "AI drafts were reviewed by the owning teammate" has not been confirmed. Each member should review their own line and correct it before submission.

| Member | Area (as drafted) |
|---|---|
| Ameya | Data pipeline, conversion, grouped split, validation |
| Prathmesh | Training pipeline, YOLOv8n and YOLO11n baselines |
| Tejas | Architecture changes (P2, attention, combined), Week 5 ablations, unit tests |
| Vrishin | Evaluation, efficiency benchmarking, error analysis, Week 6–7 packaging and reproduction |

---

## 9. Quality checks: verified status

The Week 7 checklist (`docs/week7_quality_checklist.md`) has been regenerated with the statuses below. Items previously marked "Pass" without verification are now marked accurately.

| Check | Status now | Evidence |
|---|---|---|
| Raw dataset not committed | Yes (`.gitignore`), but no git history exists to verify against | `.gitignore` |
| Test metrics not used for selection | Yes (per saved fields) | `test_used_for_selection: false`; validation-only lock |
| Final test coverage | Counts re-computed on the full split (amendment `week6_full_test_counts_v1`); original smoke outputs retained | `results/week6/amendment_full_test_counts/` |
| No personal machine paths in configs | **Fails** for `configs/rdd2022.yaml` (generated, gitignored) | `src/paths.py` |
| Official RDD2022 used for results | **Fails** (fixture) | `manifest.csv` `source_image` |
| Reproduction script | Runs, degenerate (all zeros) | `reproduction_results.json` `degenerate_all_zero: true` |
| Figures have titles and labels | Titles and labels present | Visual review, Section 3.7 |
| Figures show informative results | **Fails** (most empty or flat) | Visual review, Section 3.7 |
| README usable from a clean environment | Verified on macOS arm64: venv from `requirements.txt`, 8/8 tests, reproduction runs | Section 3.6 |
| Colab / Kaggle setup | **Not run** | — |
| Pinned dependencies | Yes (`requirements-lock.txt`) | Section 3.6 |
| AI disclosure present | Yes (`docs/AI_use.md`) | File |
| Team contributions | Present, **not verified by team members** | Section 8 |
| Git history | **None** (no commits) | `git log` |

---

## 10. Artifact index

| Area | Location |
|---|---|
| README and setup | `README.md`, `requirements.txt`, `requirements-lock.txt`, `environment.yml`, `data/README.md` |
| Data splits | `data/splits/` (`train.csv`, `val.csv`, `test.csv`, `split_summary.json`) |
| Configs | `configs/` (`rdd2022.yaml` generated, baselines, Week 5 `week5_*.yaml`, `final_evaluation.yaml`) |
| Model YAMLs | `configs/models/yolo11n-p2.yaml`, `yolo11n-attn.yaml`, `yolo11n-p2-attn.yaml` |
| Scripts | `scripts/` (see `docs/repository_audit.md`), including new `scripts/plot_modeling_set_classes.py` |
| Source code | `src/` (dataset, training, models, evaluation, paths, constants) |
| Week 3–6 results | `results/week3/`, `results/week4/`, `results/week5/`, `results/week6/` |
| Week 7 results | `results/week7/` (`environment_check.json`, `reproduction_results.csv`, `experiment_manifest.csv`) |
| Packaged final results | `results/final/`, `tables/final/`, `figures/final/` |
| Figures | `figures/week4/`, `figures/week5/`, `figures/week6/`, `figures/final/`, `figures/class_distribution_modeling_set.png` |
| Reports and docs | `docs/` (week reports, protocol, dataset card, experiment log, final summary, AI disclosure, contributions, quality checklist, reproduction report, known issues) |
| Presentation | `presentation/week7_slide_outline.md` |
| Tests | `tests/test_week5_models.py` (8 tests) |

---

## 11. Reproduction commands

```bash
cd road-damage-detection
bash scripts/setup_environment.sh          # or: python -m venv .venv && pip install -r requirements-lock.txt
python scripts/check_environment.py
python -m unittest discover -s tests -v
python scripts/reproduce_main_result.py    # degenerate on the fixture; see Section 3.6
bash scripts/run_all_checks.sh
```

Smoke pipeline (fixture or real data):

```bash
./scripts/run_week3.sh --smoke
./scripts/run_week4.sh --smoke
./scripts/run_week5.sh --smoke
./scripts/run_week6.sh --smoke   # the test run is already recorded; see Section 0.2 before re-running
```

Full-budget runs (after real RDD2022 is in `data/raw/RDD2022/`): drop `--smoke`. Do not re-run the Week 6 test evaluation. Any further test reading needs a new documented amendment.

---

## 12. AI assistance disclosure

Generative AI tools were used to brainstorm the project scope, refine the research question, organize documentation, improve grammar, and assist with code planning. The team independently reviewed the dataset information, experimental design, implementation, results, and conclusions. No experimental results or performance metrics were fabricated by AI. The team remains responsible for implementing, verifying, interpreting, and presenting all project work.

*Note for the team:* This progress report and several Week 6–7 documents were drafted with AI assistance from saved result files. Before submission, each team member should confirm the numbers against the files in `results/` and confirm the caveats in Section 0.
