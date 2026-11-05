# Current protocol boundaries

This describes the 2026-11-05 numerical snapshot, not a prospective registration. Complete parameters and analysis coverage remain in the exported protocol and field-definition worksheets.

## Tasks and partitions

COG denotes concurrent memory/EF associations; COG-FU predicts subsequent cognition using baseline measurements; ΔCOG assesses cognitive-change associations; ARM classifies intervention arms; Re-ID matches participants across visits. Paired-scan ΔCOG/ARM analyses are not pretreatment response predictions. DEMO covers age and sex; HRV-ER uses seven age categories and balanced accuracy. DX definitions are task-specific; normal cognition versus all-cause dementia and the separate AD-etiology sensitivity are not interchangeable.

Local prediction uses participant-grouped five-fold outer evaluation and an approximately 80:20 inner training/validation split, repeated across split seeds 0–2. Keep all visits/windows from a participant together. Outer stratification and inner arm stratification follow each task implementation. Scaling, PCA, regularization selection and readout selection use training partitions only. Source-fitted transfer applies a source predictor without target refitting; target-local evaluation fits a downstream predictor on target training participants. These are separate protocols.

Follow-up and paired-visit imaging probes use at most five training-fitted PCs. Sensitivity selects among 5, 10 and 20 PCs within training partitions. This includes HRV-ER single delayed recall from baseline MRI; its old 100-PC analysis is retained only as a labeled sensitivity. ADNI/NACC cognitive probes and supervised adaptation use a maximum of 100 components, capped by sample/feature dimensions. Adaptation transformations are fitted on frozen training representations and held fixed; downstream intervention PCA is fitted separately.

## Inputs and references

BrainLM uses 200 native-TR frames; Brain-JEPA 160 native-TR frames. Brain-Semantoks uses 100 frames resampled to 2 s; fMRI-LM uses 160 frames at 2 s. BrainHarmonix-F uses up to 18 native-TR temporal patches, each approximately 35.28 s. The released fMRI-LM stage-2 configurations are evaluated with brain inputs alone; this does not imply text-free pretraining. Unavailable pretraining normalization statistics are replaced by within-scan median/IQR scaling for fMRI-LM and BrainHarmonix-F.

Padding and masks are architecture-specific. BrainLM 111M/650M retain padded image patches during inference; excluding architectural padding from a reconstruction loss does not mean it was masked during inference. BrainHarmonix-F excludes fully padded tokens from valid-token pooling. Brain-Semantoks retains its missing-network treatment for unavailable cerebellar inputs. Checkpoint, atlas and preprocessing details are in Table S2.

Whole-scan Schaefer100 FC and frozen BrainGFM use identical supplied FC matrices. They are distinct from atlas-matched FC/FC-Tucker and additional actual-input-window FC controls. BrainGFM has no post-training arm. Clinical-only references and imaging-plus-clinical comparisons retain task-specific samples. HRV-ER EF clinical models use two archived category-coding protocols with reference correlations 0.730 and 0.818; neither is relabeled as the other.

## Readout and supplementary controls

Primary and alternative readouts use the same final encoder output, including final normalization where implemented. Spatial units differ by architecture. Historical BrainLM 13M layer-3 analyses are separate. Training-selected readout/ridge analyses cover ADNI/NACC cognition, with 36 configuration/outcome/cohort fits, 540 outer-fold choices and 164 paired contrasts.

Random-initialization controls cover all nine configurations and seven cohorts, using five seeds (101–105), primary and embedding-connectivity readouts, and native or fixed 128-dimensional projections. References average participant scores over random seeds, not embeddings. Temporal-order controls use five shuffles (211–215), with a common temporal permutation across regions inside observed windows. Padding and missing-input masks remain fixed. These perturbations test sensitivity to ordering, not a uniquely neural mechanism.

Regional BOLD SD is a supplementary signal control, not a main comparator. It uses model-matched observed windows before model-specific regional normalization, excluding padding. Beyond Re-ID, SD controls cover concurrent and follow-up cognition, cognitive-change associations and arm classification; all 2,696 comparisons are in Table S15.

FD-adjusted Re-ID covers nine configurations and seven cohorts. OLS with an intercept and log(1+mean FD) is fitted in 20% calibration participants and applied to the remaining 80%, with three overlapping splits (0–2). The 3,024 contrasts use four Holm families of 756 tests. The earlier pSOPT-only family is superseded for current motion inference. A separate scan-level FD regression uses primary frozen embeddings, participant-grouped evaluation and a log(1+FD) target; 112 fitted configurations/references and 133 paired comparisons are reported without new absolute-null claims.

## Statistical evidence

Main paired comparisons use 5,000 participant bootstraps and 20,000 two-sided fixed-prediction or query-score swaps. Intervals condition on fitted predictions and, for Re-ID, the fixed gallery. Absolute frozen-prediction tests refit the complete prediction pipeline under 1,999 participant-block label permutations; identity uses 20,000 mappings. Physiological circular-shift tests use 1,999 refits. Full-family Holm correction is retained. Above-chance evidence, superiority to the designated FC reference, and consistency across independent cohorts are separate criteria. Consistency requires corrected superiority in at least two independent cohorts and positive differences in every evaluable cohort for the same task/protocol; source cohorts do not count as independent post-training transfer evaluations.
