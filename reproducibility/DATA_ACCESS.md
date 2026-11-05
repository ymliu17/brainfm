# Data access and local layout

No participant-level files are included. In particular, this repository excludes regional time series, FC matrices for individual scans, cognitive/clinical labels, ECG/PPG recordings, per-participant predictions, embeddings, trained adapters, and source checkpoint weights. Aggregate tables, numerical figure inputs and rendered figures are also excluded; obtain or generate them separately for plotting.

* **ADNI:** obtain access through the [ADNI data access process](https://adni.loni.usc.edu/data-samples/access-data/).
* **NACC:** obtain data through the [National Alzheimer's Coordinating Center](https://naccdata.org/).
* **HRV-ER:** [OpenNeuro ds003823, version 1.3.5](https://openneuro.org/datasets/ds003823/versions/1.3.5). Follow the dataset's terms and consult `EXPERIMENTS.md` for the older-adult subgroup, assessment mapping, and physiology processing.
* **ACT, CogTE, BREATHE, pSOPT:** direct access requests to **Vankee Lin**. No promise of automatic access, specific approval timeframe, or unrestricted redistribution is made here; confirm conditions with the custodian.

The original code uses project-relative `data/`, `checkpoints/`, and `runs/` directories. Some legacy paths also support environment-variable overrides through `src/paths.py`; newer model modules use their own project-relative paths. Inspect the loader for the analysis you are reproducing. Dataset-specific names and schemas are part of the code and should not be replaced by a single generic matrix without checking visit and outcome alignment.

| Input group | Loader / contract |
|---|---|
| Original ADNI and intervention cohorts | `src/paths.py`, `src/cohorts.py`, `src/datasets.py` |
| NACC and HRV-ER | cohort preparation in `src/` and scripts identified in `EXPERIMENTS.md`; local directories `data/NACC/`, `data/HRV_biofeedback/` |
| Model-specific time-series inputs | `src/model_input_protocol.py`, `src/brainlm_input.py`, family-specific `data.py` and encoder modules |
| Whole-scan Schaefer100 FC | `scripts/sf100_benchmark.py` and associated loaders; ACT follow-up maps to T3_6mo, BREATHE uses run-01 |
| Physiological observations | ECG/PPG loaders and window alignment in the physiological modules; keep participant/visit/time mapping local |

For NACC, selected cross-sectional observations and repeated scans for other tasks are distinct eligibility sets. HRV-ER's delayed-recall assessment is not a repeated memory composite. BREATHE cognitive analyses exclude its nonrandomized subgroup, whereas identification coverage differs. The task loaders and aggregate sample/coverage tables document these distinctions.

Do not commit newly obtained data into this repository. `.gitignore` blocks raw-data locations and model/array formats; it is not a substitute for reviewing files before publication.
