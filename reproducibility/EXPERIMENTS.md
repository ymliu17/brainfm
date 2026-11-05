# Computational entry points

Use `config/benchmark_scope.json` for the current configuration/cohort scope. Older-named helpers are retained only when required by these entry points or by reported exploratory controls. Data eligibility and outcome definitions remain implemented by the associated loaders; no generic replacement protocol is imposed.

| Analysis | Computational entry points | Results |
|---|---|---|
| Preparation | `prepare_adni.py`, `prepare_adni_paired.py`, `prepare_nacc_parity.py`, `prepare_nacc_uds.py`, `prepare_hrv_memory.py`, `prepare_psopt.py` | S1 |
| Original four configurations | `fm_readout_v4.py`, `fm_matched_v3.py`, `fm_matched_supplement.py`, `fm_posttraining_matrix.py`, `fm_cls_posttraining_v5.py` | S3–S6, S10 |
| Brain-Semantoks / fMRI-LM VQ-contrastive | `new_fm_benchmark.py`, `src/new_fm/` | S3–S10 |
| fMRI-LM MAE / VQ-domain | `fmrilm_variant_task.py`, `src/fmrilm_variants/` | S3–S10 |
| BrainHarmonix-F | `brainharmonix_task.py`, `src/brainharmonix/` | S3–S10 |
| FC / FC-Tucker / BrainGFM | `sf100_benchmark.py`, `run_sf100_loso.py`, `run_sf100_hrv_recall_axes.py`, `src/thirdorder.py` | S3, S5, S6 |
| Common-sample comparisons | `sample_alignment.py`, `paper_completion.py` | S3–S10 |
| Paired visits | `intervention_prepost.py`, `intervention_prepost_inference.py` | S8 |
| Final-layer readouts | `run_connectivity_readouts.py`, `run_readout_task_extension.py`, `run_readout_completion.py`, `run_intervention_readouts.py` | S4 |
| Training-selected readouts | `run_nested_readout_selection.py` | S4 |
| Post-training readout controls | `run_adapted_readouts.py`, `run_adapted_readouts_focused.py` | S10 |
| Alignment | `src/embedding_harmonization.py`, `run_intervention_alignment.py` | S11 |
| PCA / input length | `run_posttraining_pca_sensitivity.py`, `run_intervention_input_length.py`, `paper_robustness.py` | S7 |
| HRV-ER five-PC recall revision | `run_hrv_recall_pca_revision.py` | S3–S7, S10–S12 |
| HRV-ER age-category comparators | `run_hrv_age_category_comparators.py` | S5 |
| Physiology | `run_physiology_matched.py`, `run_ecg_frozen_no_usable.py`, `run_new_fm_physiology.py` | S9 |
| Refitted absolute tests | `run_evidence_nulls.py` | S12 |
| Separate FC/BrainGFM null extension | `run_comparator_absolute_nulls.py` | Separate extension; not substituted for this MS snapshot |
| Nine-configuration random/signal Re-ID controls | `run_identity_controls.py`, `run_nine_model_identity_controls.py`, `src/readout_control_encoders.py` | S13 |
| Temporal-order controls | `run_temporal_order_controls.py` | S13 |
| Regional SD beyond Re-ID | `run_sd_prediction_controls.py` | S15 |
| FD intake / adjusted Re-ID | `prepare_motion_covariates.py`, `src/motion_covariates.py`, `run_all_cohort_fd_controls.py` | S1, S2, S13 |
| Scan-level FD prediction | `run_scan_embedding_fd_prediction.py` | S13 |
| Paired inference / correction | `src/paper_paired_stats.py`, `scripts/joint_inference.py`, family-specific inference | Full named families |
| Aggregate-only reproduction | `summarize_evidence.py`, `plot_results.py` | S14 and main/supplementary numerical plots |

Script paths in the table are under `scripts/` unless specified otherwise. `joint_inference.py` is a shared computational library: some legacy internal field names retain the six-model label for compatibility; the final expanded joint result uses `holm_p_joint.2`. Its obsolete six-model manuscript/report command has been removed.

## Protocol boundaries

See [CURRENT_PROTOCOLS.md](CURRENT_PROTOCOLS.md) for the current task names, five-PC recall revision, final-layer controls, SD and FD experiments. Full dataset field definitions and coverage are retained in `results/tables/`. Older-named computational helpers remain where called by the retained entry points; superseded stand-alone plotting inputs, writing utilities, logs and queue/watchdog scripts are excluded.
