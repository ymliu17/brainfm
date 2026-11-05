# Local result input layout

Numerical tables and figure source data are stored separately from GitHub. The following mapping documents the inputs expected by the plotting and evidence code; it does not indicate that the files are included.

Set `BRAINFM_RESULTS_DIR` to the local export directory (default: ignored `results/`). `tables/` contains aggregate worksheets exported as `Dataset_S<dataset>__<worksheet>.csv`; `figure_data/` contains the source CSVs described in `FIGURE_DATA.md`. Preserve the complete correction families and source precision when preparing these inputs. The separate local release's catalog records its worksheets and dimensions.

| Table | Content | Dataset |
|---|---|---|
| S1 | Cohorts, outcomes, TR, clinical and PCA protocols, sample/FD coverage | S1 |
| S2 | Configurations, checkpoints, inference families and coverage | S2 |
| S3 | Atlas- and input-window-matched FC and FC-Tucker | S3 |
| S4 | Fixed final-layer readouts, matched controls, intervention readouts, training-selected readouts | S4 |
| S5 | Primary frozen local, diagnostic, demographic and Re-ID results | S5 |
| S6 | Source-fitted transfer, LOSO and HRV-ER recall transfer | S6 |
| S7 | PCA, input context, covariates, fusion and participant weighting | S7; detailed PCA choices in S14 |
| S8 | Paired-visit cognitive-change associations and arm classification | S8 |
| S9 | Physiological analyses and temporal nulls | S9 |
| S10 | Post-training, clinical increments, checkpoints and focused adapted-readout comparisons | S10 |
| S11 | Mean/SD and covariance alignment | S11 |
| S12 | Absolute tests, cohort evidence and summary criteria | S12 |
| S13 | Random initialization, signal controls, time shuffling and FD analyses | S13 |
| S14 | Cohort-specific evidence display | S12: `S12_cohort_evidence_matrix` and `S12_cohort_evidence_tests` |
| S15 | Regional BOLD SD controls for cognition and intervention tasks | S16 |

Dataset S15 contains figure source data; it is **not** Table S15. Original worksheet names in this dataset may reflect earlier figure numbering. Table S14 is reconstructed by `scripts/summarize_evidence.py` from 369 cohort-specific adjusted tests. Its filled/open circles describe absolute evidence, not superiority to FC. The separate `criteria_counts.csv` output reports absolute support, FC superiority and cross-cohort consistency independently.

## Interpretation

Use `p_adjusted` for standardized supplementary tables. Selected figure CSVs preserve `holm_p` or the final joint field `holm_p_joint.2`; the plotting code uses the archived final values and never recalculates correction on a selected subset. Earlier corrections and historical rows are retained only where the finalized datasets label them explicitly.

HRV-ER imaging-only delayed recall uses the revised five-PC primary protocol. The 100-PC result and its absolute tests are explicitly archived sensitivities, not alternative primary results. Table S13 retains historical pSOPT-only motion contrasts; current motion inference uses `S13_FD_ReID_tests` and its seven-cohort correction. Additional coverage and exceptions are defined in the dataset `Contents`, field definitions and protocol worksheets.

The absolute-test panel in this manuscript snapshot concerns the nine primary time-series configurations. Comparator null-test extensions have separate computation entry points; their results must not silently replace the manuscript families. Physiological absolute tests assess temporal alignment of the complete covariate-plus-embedding procedure. SD and scan-average FD controls do not inherit absolute significance from positive correlations or paired tests.

With the local inputs available, the supplementary plot command creates numerical replots, with simplified layouts; it does not distribute manuscript artwork, captions or Word files.
