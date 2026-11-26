# Week 7 presentation outline (10–12 slides)

Suggested length: ~10–12 minutes. Use only verified figures/tables under `figures/final/` and `tables/final/`.

---

## Slide 1 — Title and team

- **Title:** Road Damage Detection Using Lightweight Object Detectors  
- **Content:** DATA-255 · RDD2022 · Classes D00/D10/D20/D40  
- **Team:** Ameya (data), Prathmesh (training/baselines), Tejas (architectures/ablations), Vrishin (eval/efficiency)  
- **Suggested visual:** Project logo / sample road image (no fabricated metrics)  
- **Speaker notes:** Introduce roles; state that test was frozen after val-only selection.  
- **Owner:** All (Ameya opens)

## Slide 2 — Motivation and problem

- **Content:** Manual inspection is costly; need detectors that run on modest GPUs (T4-class) for multi-country road imagery. Thin cracks and potholes are small and easy to miss.  
- **Suggested figure:** `figures/final/class_distribution.png` or a labeled sample  
- **Speaker notes:** Safety/maintenance context without claiming production readiness.  
- **Owner:** Ameya

## Slide 3 — Research question and hypotheses

- **Content:** From a small detector, which single change best improves accuracy per added cost: P2 head, lightweight neck attention, or 800 px input? Combined model as upper bound.  
- **Suggested visual:** Hypothesis bullets (H1–H3) from `docs/research_question.md`  
- **Speaker notes:** Emphasize fair controls and val-only selection.  
- **Owner:** Tejas

## Slide 4 — RDD2022 and classes

- **Content:** Four classes; China_Drone excluded; portal/Figshare links  
- **Suggested table:** Class ID table; country mix from dataset card  
- **Speaker notes:** Annotation noise and imbalance caveats.  
- **Owner:** Ameya

## Slide 5 — Preprocessing and leakage-safe split

- **Content:** VOC→YOLO; grouped 70/15/15 seed 6845; pHash duplicates same-split; negatives kept  
- **Suggested figure:** Split summary counts (`data/splits/split_summary.json`)  
- **Speaker notes:** Why grouping blocks consecutive frames.  
- **Owner:** Ameya

## Slide 6 — Detection pipeline

- **Content:** Ultralytics YOLO train/val; AMP on CUDA; checkpoint = best val mAP@0.5; metrics list  
- **Suggested diagram:** Raw → YOLO labels → train → val select → frozen test  
- **Speaker notes:** Test never used for tuning.  
- **Owner:** Prathmesh

## Slide 7 — YOLOv8n and YOLO11n baselines

- **Content:** Week 3/4 baselines @ 640; fair recipe  
- **Suggested figure/table:** `figures/final/baseline_comparison.png`, `tables/final/baseline_comparison.csv`  
- **Speaker notes:** Quote **validation** numbers for selection narrative; mention smoke caveat if mAP near zero.  
- **Owner:** Prathmesh

## Slide 8 — Week 5 ablations

- **Content:** P2 @640, attention @640, 800 px, combined P2+attn@800; seeds 0/1/2 plan  
- **Suggested figure:** `figures/final/ablation_map_comparison.png`  
- **Speaker notes:** No ablation beat YOLO11n@640 on available val mAP; seeds 1/2 Pending.  
- **Owner:** Tejas

## Slide 9 — Final test results

- **Content:** Frozen protocol; conf=0.25 (val-locked); final = YOLO11n@640; Val vs Test columns  
- **Suggested table:** `tables/final/final_comparison_table.csv` (show Test mAP50 / mAP50-95 / P / R)  
- **Speaker notes:** Do not retune after test; smoke-scale absolute values are low — be honest.  
- **Owner:** Vrishin

## Slide 10 — Error analysis and efficiency

- **Content:** FP/FN/missed small / localization; params, GFLOPs, latency  
- **Suggested figures:** `figures/final/false_negatives.png`, `missed_small_objects.png`, `accuracy_efficiency_tradeoff.png`  
- **Speaker notes:** Pothole misses and thin cracks; combined model costs more GFLOPs.  
- **Owner:** Vrishin

## Slide 11 — Limitations and future work

- **Content:** Smoke budgets; incomplete multi-seed; US-only test geography; need full-data training; optional true country-held-out track  
- **Suggested visual:** Bullet list only  
- **Speaker notes:** No unsupported SOTA claims.  
- **Owner:** Tejas + Vrishin

## Slide 12 — Conclusion and contributions

- **Content:** Selected YOLO11n@640; reproducible package; role summary; AI disclosure one-liner  
- **Suggested visual:** Contribution grid from `docs/team_contributions.md`  
- **Speaker notes:** Thank instructors; point to README reproduction commands.  
- **Owner:** All (Vrishin closes)

---

## Backup slides (optional)

- Confusion matrix / PR curves (`figures/final/`)  
- Country table (`tables/final/country_level_results.csv`)  
- Reproduction pass/fail (`results/week7/reproduction_results.csv`)
