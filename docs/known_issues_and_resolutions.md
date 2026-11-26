# Known issues and resolutions

**Last updated:** 2026-11-26 (official RDD2022 integrated; see section H)  
**Purpose:** one place that lists every problem found in the project audit, the evidence for it, what was changed, and what still needs action. Status values:

- **Resolved**: fixed in code or docs, and verified.
- **Resolved (docs/caveat)**: the problem cannot be fixed in code here; it is now stated accurately everywhere it appears.
- **Open: decision**: needs a team or instructor decision before it can be closed.
- **Open: blocked**: needs real data, a GPU, or user action that is not available in this environment.

---

## A. Data provenance and validity

| ID | Issue | Evidence | Status | Action taken | Remaining action |
|---|---|---|---|---|---|
| A1 | **All Weeks 3–7 results use a synthetic fixture, not RDD2022.** | `data/processed/RDD2022_yolo/manifest.csv` `source_image` points to `data/fixture/RDD2022/`. | Data: **Resolved** (2026-11-26). Results: **Open: blocked** (needs GPU training) | Official RDD2022 downloaded, audited, converted, split and validated (section H). The active pipeline now defaults to official data; fixture results are marked LEGACY (`results/LEGACY_FIXTURE_RESULTS.md`). | Train and evaluate on official data (F2–F4). |
| A2 | Week 5 and Week 6 reports described fixture-based results as if RDD2022. | `docs/week5_ablation_report.md` said "limited RDD2022 subsets". | Resolved | Wording corrected in `scripts/write_week5_decision.py` and regenerated. | None for this item. |
| A3 | Class-distribution figure described the wrong set. | `figures/class_distribution.png` and the packaged copy show 108/123/110/102 (443 boxes). These are Week 1 audit counts on raw fixture XMLs, including China_Drone. The modeling set has 93/105/103/87 (388 boxes). | Resolved | New `scripts/plot_modeling_set_classes.py` writes `figures/class_distribution_modeling_set.png` from `results/week3/dataset_validation.json`. The packaging script now copies that figure, and the mislabeled copy was removed from `figures/final/`. | None. The audit figure `figures/class_distribution.png` is still the raw-audit figure; its title should not be cited as the modeling set. |
| A4 | Audit and conversion unsupported-label counts differ. | Audit: D50 104, Repair 84 (`results/audit_summary.json`). Conversion: D50 94, Repair 75 (Week 3 notes). | Open (minor, not verified) | None. | Likely because the audit counts include China_Drone, which conversion excludes. Confirm by counting per country. |

---

## B. Final test evaluation (Week 6)

| ID | Issue | Evidence | Status | Action taken | Remaining action |
|---|---|---|---|---|---|
| B1 | **The final test evaluation was run in smoke mode.** The one-shot test protocol assumed full test coverage. | `results/week6/final_run_manifest.json` has `"smoke": true`. In `scripts/run_final_evaluation.py`, `count_fp_fn(..., max_images=12 if args.smoke else None)` limited FP/FN counts and error analysis to the first 12 sorted test images. | **Resolved by amendment `week6_full_test_counts_v1`** (decision: re-count on full split) | `scripts/amend_full_test_counts.py` re-read the full 50-image test split once, with the same frozen checkpoint (`yolo11n_640` seed 0), operating confidence 0.25 (validation-locked), NMS IoU 0.7, and matching IoU 0.5. It did not retrain and changed no threshold or selection. Record: `results/week6/amendment_full_test_counts/full_test_counts.json` (`model_changed`, `thresholds_changed`, `selection_changed`, `retrained` all false). The original smoke-run outputs are kept (`results/week6/final_test_results.csv`), and the error analysis originals are archived in `results/week6/archive_smoke_12img/`. The report shows both. | Disclose in any write-up that the test split was read a second time for counts after the original results were observed. Full-split numbers: 50 images, 55 GT, 219 FP, 42 FN, 13 localization-poor, 0 TP. Error analysis on the full split: 274 errors, which reconciles with 219 + 42 + 13. |
| B2 | Country table showed 8 images for United_States; split has 50. | `results/week6/country_results.csv`: 8 images, 11 objects (smoke cap). `data/splits/split_summary.json` test: 50 images, 55 boxes. | Resolved | Cause: `country_metrics` applies `limit = limit[:8]` when `smoke` is set. The amendment wrote full-split country rows (`full_test_country_results.csv`: 50 images, 55 objects). The report uses those rows. | None. |
| B3 | Test FP/FN counts for the final model: 72 FP / 11 FN in the smoke run. | `final_test_results.csv`. These come from 12 images. | Resolved by amendment | Full-split counts (219 FP, 42 FN, 0 TP) are in the report. The smoke counts are kept and labeled as such. | None. |
| B4 | The confidence threshold 0.25 was chosen when every threshold gave F1 = 0. | `results/week6/CONFIDENCE_THRESHOLD_LOCKED.json`: six thresholds, all F1 0.0, criterion `max_f1_then_precision`. | Resolved (disclosed) | Disclosed in the progress report and Week 6 report. | None until real data gives a non-degenerate sweep. |
| B5 | Efficiency benchmark is on Apple MPS, not the T4 the protocol requires. | `results/week6/efficiency_benchmark.csv`: `is_t4 = False`, `device = mps`. | Resolved (labeled); **Open: blocked** for T4 numbers | Labels kept as they are. | Run `scripts/final_efficiency_benchmark.py` on a Colab or Kaggle T4 and record that as a new result. |

**Decision (B1), recorded:** the team chose option (2), a full-split recount recorded as a protocol amendment. It was applied as described in B1. The other options were not taken.

Options that were considered:

- **(1)** Keep the current results and state the limitation.
- **(2) Chosen.** Re-count the full 50-image test split with the same frozen checkpoint, threshold, and IoU, recorded as a versioned amendment.
- **(3)** Treat the test split as consumed and start a new protocol version when real RDD2022 is available.

---

## C. Reproducibility and checks

| ID | Issue | Evidence | Status | Action taken | Remaining action |
|---|---|---|---|---|---|
| C1 | **Reproduction PASS is trivial.** All compared values are 0.0. | `results/week7/reproduction_results.csv` before this change: 9 of 9 PASS, all 0.0. | Resolved (flagged) | `scripts/reproduce_main_result.py` now detects an all-zero comparison, prints a warning, writes `degenerate_all_zero: true` to JSON, and puts a warning at the top of `docs/reproduction_report.md`. Re-run completed with the flag. | A non-degenerate reproduction needs real data. |
| C2 | Week 7 checklist marked items "Pass" that were not verified. | Original `scripts/write_week7_docs.py` hardcoded "Pass" for figures, clean-environment README, team contributions, and others. | Resolved | Checklist rows now show the actual status. Reproduction row is computed from `reproduction_results.csv`. | None for the checklist itself. Items still unverified are listed as such. |
| C3 | Dataset YAML "fixed" claim was false. | `write_week7_docs.py` said `rdd2022.yaml` was changed to a relative path. `src/paths.py` writes an absolute path on purpose. | Resolved | Claim corrected in `docs/repository_audit.md` and the checklist. `configs/rdd2022.yaml` added to `.gitignore` because it is machine-generated. | None. A portable path is not possible with Ultralytics; the file is regenerated per machine by `check_environment.py` or `setup_environment.sh`. |
| C4 | **Clean-environment install and tests not verified.** | No prior clean install. | Resolved | Created a fresh venv from `requirements.txt` on macOS (arm64). `python -m unittest discover -s tests -v`: **8 of 8 pass**. Reproduction script ran in that environment. | Run the same steps on Colab or Kaggle (see D1). |
| C5 | Dependencies were not pinned. Earlier results used Ultralytics 8.4.189 and torch 2.14.1; the clean install gets 8.4.219 and torch 2.15.0. | `results/week6/final_run_manifest.json` software block vs. clean-environment `pip freeze`. | Resolved | `requirements-lock.txt` created from the verified clean environment (52 packages, torch 2.15.0, ultralytics 8.4.219, numpy 1.26.4, pandas 3.0.6). | Note the version drift in any write-up. Re-running with the lock may change the numerics slightly. |
| C6 | Project `.venv` in the workspace has no torch installed. | `.venv/bin/python -c "import torch"` fails with `ModuleNotFoundError`. | Open (user action) | None. This environment was not modified. | Recreate with `bash scripts/setup_environment.sh` or use the clean venv pattern from `requirements-lock.txt`. |
| C7 | Model checkpoints (`*.pt`) are gitignored, so teammates cannot reproduce from the repo alone. | `.gitignore` `*.pt`. | Open (by design) | Documented in the Week 7 docs. | Share checkpoints through a shared drive, as the Week 7 docs suggest. |
| C8 | Absolute machine paths in result CSV and JSON files. | `results/` files contain `/Users/ameyakhond/...`. | Resolved (caveat) | Accepted as run logs. Not used as configs. | None. |
| C9 | No git commits exist. | `git log`: "branch 'main' does not have any commits yet". | Resolved by decision (no commit now) | None. The team decided not to commit yet. | Commit when ready. The `.gitignore` excludes `data/raw/**`, `data/processed/**`, `*.pt`, `.venv`, and `configs/rdd2022.yaml`. |

---

## D. Figures (visual review of all 39 PNGs)

All figures were reviewed by building contact sheets and inspecting them. Findings:

| Figure group | Finding | Status |
|---|---|---|
| `figures/final/precision_recall_curves.png`, `figures/week6/precision_recall_curves.png`, `figures/week4/*_BoxPR_curve.png` | Empty: all classes 0.000 AP; curve is flat at zero. | Resolved (caveat). Not informative until real data. |
| `figures/week4/*confidence_analysis.png`, `figures/week4/*confidence_analysis`, `YOLOv8n` / `YOLO11n` sweeps | Flat at 0 for precision, recall, and mAP@0.5 across all thresholds. | Resolved (caveat). |
| `figures/final/country_performance.png`, `figures/week6/country_performance.png` | Empty: bars have no values (Pending or 0 proxy recall). | Resolved (caveat). |
| `figures/week5/training_curves.png` | Empty: 1 epoch. | Resolved (caveat). |
| `figures/week6/confusion_matrix_tp_only.png` | All cells 0; colorbar runs -0.1 to 0.1. Not informative. | Resolved (caveat). |
| `figures/week5/*`, `figures/final/ablation_map_comparison.png` | Single non-zero bar (baseline, ~8e-6) and zero bars for the rest. Axis scale shows 1e-6 units. | Resolved (caveat). |
| `figures/final/false_positives.png`, `false_negatives.png`, `missed_small_objects.png`, `localization_errors.png`, `figures/week6/*` | Qualitative galleries show random coloured noise with box overlays: the synthetic fixture images are not road scenes. | Resolved (caveat). Do **not** use these in a presentation as evidence of road damage detection. |
| `figures/week4/*confusion_matrix.png`, `figures/final/confusion_matrix.png` | Dominated by "background" column counts (thousands). Labels readable. | Resolved (caveat). |
| `figures/class_distribution_modeling_set.png` (new) | Correct and labeled (see A3). | Resolved |
| Titles and axis labels | Present on all figures reviewed. Legends are small on the Week 5 and Week 6 figures. | Resolved for titles and labels |

**Recommendation:** rebuild all figures on real RDD2022 results. Do not use any of the current figures in the presentation as evidence of model performance.

---

## E. Team and documentation

| ID | Issue | Evidence | Status | Action taken | Remaining action |
|---|---|---|---|---|---|
| E1 | Team contributions were AI-drafted and claimed "AI drafts reviewed by the owning teammate." | `docs/team_contributions.md` text. | Resolved (wording) | Generator now says the section is not verified by team members. | **Each member must confirm or correct their own section.** |
| E2 | AI disclosure present. | `docs/AI_use.md`. | Resolved | Contains the required text. | None. |
| E3 | Progress report made claims that needed correction. | `docs/progress_report.md`. | Resolved | Updated with B1 to B5, C1 to C9, and the figure review. | None. |

---

## F. Items that need real data, a GPU, or the team

These cannot be closed in this environment and are not resolved.

| ID | Item | Why it is blocked |
|---|---|---|
| F1 | Official RDD2022 data run | **Data resolved:** `scripts/download_rdd2022.py` fetches the archive and the pipeline prepares it (section H). Model training on it is still F2. |
| F2 | 100-epoch training with early stopping (YOLOv8n, YOLO11n, five Week 5 variants) | Needs substantial GPU compute (25,355 training images). Only a 2-epoch, 300-image official smoke test has run (pipeline check, not performance). |
| F3 | Week 5 seeds 1 and 2 | Not run. Needs compute and the full protocol. |
| F4 | T4 efficiency benchmark | Needs a T4 GPU (Colab or Kaggle). The MPS numbers are not T4 numbers. |
| F5 | Colab setup run | Not executed. Needs Google Drive with the dataset. |
| F6 | Re-selection on validation, if the ranking changes | Only after F1 and F2. Any change to the selected model after test results have been seen must be a documented protocol amendment. |

---

## G. Items the user must do

1. ~~Answer the B1 decision.~~ **Done:** full-split recount recorded as amendment `week6_full_test_counts_v1`.
2. Initial git commit: **the team chose not to commit yet** (C9). Commit when ready.
3. Confirm or correct each team member's section in `docs/team_contributions.md` (E1).
4. ~~Obtain RDD2022 (F1).~~ **Done** 2026-11-26 (section H). Run official training on a T4 (F2).
5. Recreate the workspace `.venv` or use the clean venv pattern (C6).

---

## H. Official RDD2022 integration (2026-11-26)

| ID | Issue | Evidence | Status | Action taken | Remaining action |
|---|---|---|---|---|---|
| H1 | Official S3 archive URL returns HTTP 403. | `download_rdd2022.py` log. | Resolved | Falls back to the Figshare mirror linked from the same README (13,264,172,619 bytes). | None. |
| H2 | One invalid box (`Japan_001265`, D20, xmin = xmax = 198). | `results/official_rdd2022/conversion_issues.csv` | Resolved | Rejected and logged; the image keeps its other boxes. | None. |
| H3 | Label `D0w0` (1 object) is probably a typo for D00. | `unsupported_labels.csv` | Resolved (docs) | Dropped under the strict 4-class mapping, counted. | None unless the team wants to remap it. |
| H4 | Consecutive dashcam frames are near-duplicates (pHash distance 1–4) and crossed splits (717 pairs before the fix). | `split_leakage_report.json`; visual check of sample pairs. | Resolved | Transitive merging collapsed whole countries into train, so a non-transitive purge drops the val/test member of each crossing pair (448 images, listed in `data/splits/near_duplicate_purged.csv`). 0 crossing pairs at distance ≤ 4 remain. | 872 pairs at distance 6 still cross; judged to be similar scenes, not repeated frames. Revisit if the team wants a stricter threshold. |
| H5 | Weeks 3–7 scripts and wrappers defaulted to fixture paths. | `src/paths.py` history. | Resolved | Training scripts default to official data; legacy evaluation scripts use explicit fixture paths; `run_week3-6.sh` require `ALLOW_LEGACY_FIXTURE=1`. | Port Week 6 evaluation to official data after training. |
| H6 | `configs/rdd2022.yaml` held an absolute Mac path and was gitignored. | Previous file contents. | Resolved | Now portable and tracked; the absolute-path Ultralytics YAML is generated per machine under `data/processed/ultralytics/`. | None. |
| H7 | Official data has 39.3% negative images and very different class mixes by country (US has no negatives; India is pothole-heavy). | `audit_summary.json` | Open (analysis) | Documented in `docs/dataset_card.md`. | Consider per-country metrics in the official evaluation. |
