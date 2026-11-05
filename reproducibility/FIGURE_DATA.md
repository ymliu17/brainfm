# Figure data and numerical replots

Numerical source files are not distributed in this repository. Place them in a local result directory and set `BRAINFM_RESULTS_DIR` before running the commands below.

`python scripts/plot_results.py` renders current Figures 2–4. Add `--supplementary` to render numerical replots of all eleven supplementary result sets. These commands require separately held aggregate CSV files, do not fit models, and do not alter correction families. Main-panel organization follows the current figures; supplementary layouts are simplified numerical replots, not copies of the typeset manuscript artwork.

The original CSV names are stable provenance identifiers and sometimes contain **old figure numbers**. Use this mapping:

| Current figure | Files in `results/figure_data/` (omit `.csv`) |
|---|---|
| Fig. 2 | `Figure2_heatmap_scores_and_tests` |
| Fig. 3A–B | `Figure5A_cognitive_readouts` |
| Fig. 3C–D | `Figure4C_baseline_imaging_followup_increment` |
| Fig. 3E–F | `Figure4D_longitudinal_imaging_change_increment` |
| Fig. 4A–B | `Figure5C_within_source_posttraining` |
| Fig. 4C–D | `Figure5D_external_posttraining` |
| Fig. 4E–F | `FigureS3_all_alignment` |
| Fig. S1 | Dataset S15 worksheet `S1_relocated_radar`; `Figure2_scores`, `Figure2_ReID_scores` |
| Fig. S2 | Dataset S15 worksheet `S1_absolute_performance`; full tests in Dataset S12 |
| Fig. S3A–B | `Figure3_AB_scores` |
| Fig. S3C, D, F | `Figure3_CDF_paired_tests` (`figure_panel` is the **current** C/D/F mapping) |
| Fig. S3E | `Figure3_E_counts` |
| Fig. S4 | Dataset S5 worksheet `S5_identification_scores` |
| Fig. S5 | `Figure3_source_transfer`, `Figure3_LOSO_scores`, `Figure3_LOSO_paired` |
| Fig. S6 | `Figure4A_arm_scores`, `Figure4B_cognitive_change_input_differences`, `Figure4C_baseline_imaging_followup_increment`, `Figure4D_longitudinal_imaging_change_increment` |
| Fig. S7 | `FigureS2_frozen_scores`, `FigureS2_input_paired`, `FigureS2_clinical_increment` |
| Fig. S8 | `FigureS6_all_primary_source_transfers`, `FigureS6_all_focused_readout_tests` |
| Fig. S9 | `FigureS3_all_alignment`, `FigureS3_all_intervention_alignment` |
| Fig. S10 | `FigureS4_cognition_short_context`, `FigureS4_all_paired_short_context` |
| Fig. S11 | `FigureS5_all_physiology` |

Fig. 2 colors are **absolute score differences**, method minus matched Schaefer100 FC, not percentages. Positive differences use warm colors; negative differences use cool colors. Actual FC scores appear in the FC row. Hatching denotes unavailable outcomes. No above-chance dots are drawn. Different metrics are not put on a common effect-size scale. ΔCOG and ARM use imaging-only follow-up-minus-baseline representations, constructed before training-fitted scaling/PCA. HRV-ER DEMO age uses categorical balanced accuracy.

Fig. S3D is temporal shuffling; S3F is the pSOPT example from the **expanded seven-cohort FD correction**. The earlier pSOPT-only inference remains explicitly historical in Dataset S13.

Fig. 3 includes two HRV-ER EF clinical references on the same 55 participants and partitions: † omits a reference category in categorical coding ; ‡ retains an indicator for each category, fitted within training partitions . Both use baseline EF, age category, sex and intervention arm. The distinction is retained rather than replacing one clinical score with the other.

Across forest/cloud plots, stars mark archived corrected paired differences, **not** significant absolute performance. Intervals condition on fitted predictions; Re-ID intervals also condition on the gallery. Random/time-shuffle references average participant scores across five realizations. Full tests, negative results and family definitions are in the catalogued datasets.

Generated outputs and their source-file SHA-256 records are saved under ignored `outputs/figures/`. Figure 1 is a study-design illustration, not a numerical analysis, and is outside this computational repository.
