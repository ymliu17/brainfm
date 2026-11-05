# brainfm

Reproducible evaluation of **nine configurations from five regional rs-fMRI foundation-model families across seven aging cohorts**.

Repository: [ymliu17/brainfm](https://github.com/ymliu17/brainfm) (currently private).

This repository contains computational code, configurations, tests, model provenance, and reproduction instructions. Manuscripts, Word documents, presentation files, rendered publication assets, writing utilities, workstation logs, and obsolete scheduling scripts are excluded.

Code snapshot: **2026-11-05**. Study data, aggregate numerical tables, figure source data and rendered figures are maintained separately and are not distributed in this repository.

## Models and data

| Family | Evaluated configurations | Primary scan-level representation |
|---|---|---|
| BrainLM | 13M, 111M, 650M | Final-layer CLS, 512 / 768 / 1,280 dimensions |
| Brain-JEPA | Approximately 85M | Final-layer token mean, 768 dimensions |
| Brain-Semantoks | Teacher encoder, approximately 63M | Final-layer CLS, 768 dimensions |
| fMRI-LM | VQ-contrastive, MAE, VQ-domain; approximately 684M each | Stage-2 brain-token mean, 1,024 dimensions |
| BrainHarmonix-F | Functional encoder, approximately 89M | Valid-token mean, 768 dimensions |

Cohorts: ADNI, NACC, ACT, CogTE, BREATHE, pSOPT, and HRV-ER. Comparators include atlas-matched FC and FC-Tucker, whole-scan Schaefer100 FC, frozen BrainGFM, and clinical predictors for the relevant tasks.

## Installation and code verification

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-figures.txt
python scripts/verify_release.py
```

Full experiments require separately obtained cohort data, pretrained checkpoints and the additional dependencies in `requirements.txt`. Follow [the experimental workflow](reproducibility/REPRODUCING.md).

## Plotting separately held results

Plotting and evidence-summary code is retained, but its numerical inputs are **not included**. After generating or obtaining the corresponding result exports locally, set `BRAINFM_RESULTS_DIR` to a directory containing `tables/` and `figure_data/`:

```bash
export BRAINFM_RESULTS_DIR=/path/to/local/results
python scripts/summarize_evidence.py
python scripts/plot_results.py --supplementary
```

The default input directory is ignored `results/`. Generated outputs are saved under ignored `outputs/`. These commands do not refit models or recompute statistical tests. See [the input layout](reproducibility/RESULTS.md) and [figure-source mapping](reproducibility/FIGURE_DATA.md).

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | Encoders, input construction, adaptation, prediction, and statistical implementations |
| `scripts/` | Required preparation/evaluation entry points, shared computational helpers, and result plotting |
| `config/` | Current benchmark scope and comparator configuration |
| `reproducibility/` | Experiment map, data access, pinned model provenance, and verification manifests |
| `tests/` | Tests for the retained statistical and experimental implementations |

* [Reproduction instructions](reproducibility/REPRODUCING.md)
* [Experiment entry points and protocol boundaries](reproducibility/EXPERIMENTS.md)
* [Data access](reproducibility/DATA_ACCESS.md)
* [Result files and interpretation](reproducibility/RESULTS.md)
* [Figure-to-source mapping](reproducibility/FIGURE_DATA.md)
* [Current protocol details](reproducibility/CURRENT_PROTOCOLS.md)
* [Original models and attribution](reproducibility/THIRD_PARTY.md)

## Full experiments and tests

Full experiments require authorized cohort data and separately downloaded model weights. Install an appropriate PyTorch build for your CPU/CUDA environment, then install `requirements.txt`. The original host's distribution inventory is recorded in `reproducibility/environment-snapshot.txt`; it is not a portable CUDA lock file.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests
```

The retained code follows the dependency graph in `reproducibility/code_dependencies.json`. Some older-named modules remain because current analyses call their computational functions or because they implement explicitly retained historical controls. Model identifiers and original outcome definitions have not been renamed during cleanup.

No cohort data, numerical results, figure source data, individual predictions, embeddings, or weights are distributed. The public atlas-coordinate file under `src/toolkit/atlases/` is retained as a model dependency. Requests for ACT, CogTE, BREATHE, and pSOPT data should be directed to Vankee Lin, subject to custodian-confirmed access conditions. No repository-wide redistribution license or archived release DOI has yet been assigned; upstream terms remain applicable.
