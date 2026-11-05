# Reproducing the benchmark

## Plotting separately held aggregate results

Numerical results are not bundled with the code. First generate or obtain the corresponding aggregate exports locally. Set `BRAINFM_RESULTS_DIR` to their directory, which must contain `tables/` and `figure_data/` as described in `RESULTS.md`. Install `requirements-figures.txt`, then run:

```bash
export BRAINFM_RESULTS_DIR=/path/to/local/results
python scripts/summarize_evidence.py
python scripts/plot_results.py --supplementary
```

The evidence command reconstructs the evidence matrix and criterion counts from completed aggregate absolute tests and paired comparisons. The plotting command renders Figures 2–4 and supplementary analytical plots with their numerical source tables. Neither command requires a manuscript, an Office library, Poppler, pretrained weights, or a GPU. Output goes to `outputs/evidence/` and `outputs/figures/`.

The locally supplied corrections are retained. Filtering to displayed configurations does not redefine the comparison family. The numerical archive and each row's protocol/release fields determine which contrasts are comparable.

## Tests

Install PyTorch for the intended platform, followed by `requirements.txt`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests
```

Tests cover participant isolation, training-only transformations, null refitting, paired inference, shared samples, and intervention/readout/calibration helpers. Tests requiring existing private experiment artifacts are skipped when those artifacts are absent. These checks do not repeat source-model training or substitute for checking private-data eligibility.

## Full experimental workflow

1. Obtain authorized data through the custodians listed in `DATA_ACCESS.md` and construct the loader-specific local data layout.
2. Run `python scripts/fetch_upstream_sources.py` to obtain the pinned external source trees. Existing checkouts are not overwritten.
3. Download the specified pretrained checkpoints. Metadata and original configuration files are retained in `model_assets/` and `model_configs/`. `fetch_new_fm_assets.py` and `fetch_fmrilm_variants.py` use pinned provider metadata and verify downloaded assets. Other model checkpoint sources are documented in `THIRD_PARTY.md` and their provenance files.
4. Prepare task-specific eligibility, labels, paired-visit mappings, and model inputs. Preserve original subject identifiers locally to group repeated observations. Consult `EXPERIMENTS.md` for entry points.
5. Extract frozen representations, fit/evaluate downstream prediction models, and compute matched comparisons. Fit ADNI and NACC adaptation separately; repeat extraction and evaluation for the selected adapted checkpoints.
6. Reconstruct the full comparison families before applying joint correction. Use the complete nine-configuration aggregate outputs for comparison with this snapshot.
7. Run the refitted absolute-performance tests separately from conditional paired swaps. The former refit supervised transformations and heads; the latter condition on stored predictions.

Examples requiring private inputs and downloaded weights:

```bash
python scripts/new_fm_benchmark.py features --model semantoks --device cuda
python scripts/new_fm_benchmark.py features --model fmrilm --device cuda
python scripts/fmrilm_variant_task.py --variant mae features --device cuda
python scripts/fmrilm_variant_task.py --variant vq-domain features --device cuda
python scripts/brainharmonix_task.py features --device cuda
python scripts/new_fm_benchmark.py supervised --model semantoks \
  --source adni --target MEM --objective supervised --seed 42 --mode external --device cuda
```

These commands are individual stages. The repository intentionally excludes the original workstation's global scheduling, restart, shutdown, and manuscript-assembly services. Runtime data, logs, and computational reports are generated locally under ignored directories. Full private-data preprocessing and GPU training have not been rerun as part of repository cleanup.

## Integrity checks

Run `python scripts/verify_release.py` to verify the code/configuration manifest and check that study result files are not tracked. `reproducibility/release_manifest.json` hashes the distributed code and documentation. Numerical inputs are intentionally outside this manifest. `validation.json` records code verification; it does not imply that result data are distributed.

## Current supplemental computations

Create the ignored output roots with `mkdir -p runs outputs results/generated` before full experimental runs. After preparing the authorized data and required feature caches, use the entry points in `EXPERIMENTS.md`. Read each command’s `--help` before execution. A typical dependency order is final-layer extraction → original Re-ID pair construction → nine-configuration controls / temporal shuffling → SD features → FD intake and downstream controls. These commands are resumable and may be expensive; the aggregate-only commands above do not execute them.

```bash
python scripts/run_nested_readout_selection.py --help
python scripts/run_nine_model_identity_controls.py --help
python scripts/run_temporal_order_controls.py --help
python scripts/run_sd_prediction_controls.py --help
python scripts/run_hrv_recall_pca_revision.py --help
python scripts/run_all_cohort_fd_controls.py --help
python scripts/run_scan_embedding_fd_prediction.py --help
```

Computational exports are redirected to ignored `results/generated/`, never the manuscript directory. All raw-data and feature-cache paths remain local. Do not run a full experiment until its data-access and checkpoint requirements are met.
