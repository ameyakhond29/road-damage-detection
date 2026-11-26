# Week 7 quality checklist

| Check | Status |
|---|---|
| Scripts have usage (`argparse` / shell headers) | Not audited script-by-script |
| Paths relative or via `src/paths.py` | Mitigated: `rdd2022.yaml` is absolute by necessity (Ultralytics), machine-generated, gitignored |
| No personal machine paths in **configs** | **Fails** for `configs/rdd2022.yaml` (generated per machine; gitignored) |
| Historical absolute paths in result CSV/JSON | Present; accepted as run logs, not used as configs |
| Raw dataset not committed | Yes (`.gitignore` excludes `data/raw/**`); no git commits exist yet to verify against history |
| Test metrics not used for selection | Yes per saved fields (`test_used_for_selection: false`) |
| Final test evaluation coverage | Original run used `--smoke` (FP/FN and error analysis on 12 of 50 images). **Re-computed on the full split** under amendment `week6_full_test_counts_v1`; original retained. See `docs/known_issues_and_resolutions.md`. |
| Results traceable to experiment IDs | Yes (`experiment_manifest.csv`) |
| Tables match saved CSV/JSON | Yes for packaged copies of saved files |
| Figures have titles/labels | Not verified by visual review at time of generation |
| README usable from clean env | Not verified at time of generation |
| Reproduction script runs | Runs, but DEGENERATE: every compared value is 0.0, so this is not evidence of reproduction |
| AI disclosure present | Yes (`docs/AI_use.md`) |
| Team contributions present | Present but **not verified by team members** |
| Dataset links present | Yes (`data/README.md`); the data itself was never downloaded here |
| Official RDD2022 data used for results | **Fails**: all results use the synthetic fixture |

## Commands

```bash
bash scripts/setup_environment.sh
python scripts/check_environment.py
bash scripts/run_all_checks.sh
```
