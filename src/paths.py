"""Repository-relative paths, the cohort table, and each cohort's temporal geometry.

Layout of `data/` (see data/README.md for provenance and dates):

    data/adni/raw/           per-session ADNI scans, [424, T], amplitude intact   <- source
    data/adni/clinical/      ADNI clinical / cognitive CSVs         <- source
    data/rct/<cohort>/       bold_T1T2.pt, subjects.txt, labels.csv <- source
    data/derived/            anything a script builds (regenerable)

`BRAINLM_DATA` relocates the whole tree; the per-cohort entries below are derived from
it, so a mirror only has to reproduce the same subdirectory names.
"""
from __future__ import annotations

import os
from pathlib import Path

from temporal import COHORT_RESAMPLE, COHORT_TR

ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT / "src"
DATA_DIR = Path(os.environ.get("BRAINLM_DATA", ROOT / "data"))
RCT_DIR = DATA_DIR / "rct"
ADNI_DIR = DATA_DIR / "adni"
#: The working ADNI release (2026-08-24). Carries between-ROI amplitude, which BrainLM's
#: pretraining distribution has. A per-ROI z-scored variant -- the paired control in which
#: only amplitude differs -- is produced from this same release by
#: `prepare_adni.py --amplitude zscore`; the two agree to 7.2e-07 after z-scoring, so no
#: separate z-scored release needs to be stored.
#:
#: The superseded `denoise-old` release was deleted on 2026-08-25. It was a strict function
#: of this one (z-scoring this reproduced it to 1.9e-6) and had every per-ROI sd at exactly
#: 1.000, i.e. amplitude removed -- a measurable loss, since the amplitude profile alone
#: out-predicts the 13M encoder on CN/AD (AUC 0.682 vs 0.630).
ADNI_SCAN_DIR = Path(os.environ.get("BRAINLM_ADNI_SCANS", ADNI_DIR / "raw"))
ADNI_SCAN_DIR_RAW = ADNI_SCAN_DIR
ADNI_CLINICAL_DIR = ADNI_DIR / "clinical"
DERIVED_DIR = Path(os.environ.get("BRAINLM_DERIVED", DATA_DIR / "derived"))

CKPT_DIR = Path(os.environ.get("BRAINLM_CKPT", ROOT / "checkpoints"))
PRETRAINED_DIR = CKPT_DIR / "pretrained"
RUNS_DIR = Path(os.environ.get("BRAINLM_RUNS", ROOT / "runs"))
CACHE_DIR = Path(os.environ.get("BRAINLM_CACHE", ROOT / "runs" / "feature_cache"))
LOG_DIR = Path(os.environ.get("BRAINLM_LOG", ROOT / "logs"))

XYZ_PATH = SRC_DIR / "toolkit" / "atlases" / "A424_Coordinates.dat"

#: Centroids for the 450-ROI Schaefer-400 + Tian-S3 build, same `index x y z` MNI format.
#: Rows 1-400 are Schaefer's shipped FSLMNI152 centroids; rows 401-450 are computed from
#: the official Tian_Subcortex_S3_3T_1mm volume (yetianmed/subcortex), whose label file is
#: byte-identical to the one in this directory, so volume label i is row i. Validated by
#: hemisphere sign: all 25 `-rh` parcels come out x > 0 (min +8.2) and all 25 `-lh` x < 0
#: (max -5.2).
#:
#: SCOPE: this table exists for Brain-JEPA, which cannot use any other parcellation (its
#: position embedding is 4500 = 450 ROIs x 10 time patches, fixed in the weights). BrainLM
#: is deliberately NOT run on it. BrainLM's ROI count is set by the xyz table rather than
#: by the weights, so it *would* run -- but its pretraining and every prior result in this
#: project are on A424, and switching parcellation would be an unvalidated deviation that
#: confounds the model comparison with an atlas change. Each model is compared against
#: FC / tucker-FC computed on the SAME input that model consumes, which keeps every
#: comparison atlas-matched without moving any model off its native parcellation.
XYZ_SCHAEFER450 = SRC_DIR / "toolkit" / "atlases" / "Schaefer400Tian_S3_Coordinates.dat"

#: The released BrainLM 13M -- the warm start for stage-1 and the un-adapted comparison
#: backbone. Every entrypoint requires a checkpoint explicitly; there is no default.
OFFICIAL_BRAINLM_13M = PRETRAINED_DIR / "brainlm_13m_official" / "pytorch_model.bin"

#: The larger BrainLM releases. Same family name, different architecture: these are
#: ViTMAEForPreTraining with 2D patch embeddings, so they need the upstream padding wrapper
#: and ROI reordering rather than this tree's BrainLM path. See checkpoints/pretrained/README.md.
BRAINLM_111M = PRETRAINED_DIR / "brainlm_vitmae_111M" / "pytorch_model.bin"
BRAINLM_650M = PRETRAINED_DIR / "brainlm_vitmae_650M" / "pytorch_model.bin"

#: Official Brain-JEPA assets. Not usable with this data: Brain-JEPA expects 450 ROIs
#: (Schaefer-400 + Tian-S3) and everything here is A424. Only ViT-B is publicly released.
BRAINJEPA_DIR = PRETRAINED_DIR / "brainjepa"

# Primary ADNI clinical tables. `FILE_NAME` (the session stem) lives only in the older
# file; the diagnosis, cognition and demographic columns live only in the `_new` one.
# scripts/prepare_adni.py joins them positionally after asserting they agree row-for-row.
ADNI_COG_CSV = ADNI_CLINICAL_DIR / "ADNI_UMNPSYCH_COG_forXL.csv"
ADNI_COG_NEW_CSV = ADNI_CLINICAL_DIR / "ADNI_UMNPSYCH_COG_forXL_new.csv"

# --- Cohort table -----------------------------------------------------------------
#
# Every cohort directory holds `bold_T1T2.pt` [N, 424, T, 2] and `subjects.txt` with N
# lines in tensor-row order. `labels.csv` carries the outcome columns and the fold
# assignments, keyed by `id_col`; it does NOT have to cover every scanned subject, and
# it may list subjects that were dropped from imaging.
#
# OUTCOME SIGN: `MEM_decline_bi` / `EF_decline_bi` are named "decline" but encode
# IMPROVEMENT -- label 1 is the group whose cognition went UP over the trial (verified in
# both cohorts; see cohorts.Cohort.labels). Any text describing them as decline has the
# interpretation inverted, and AUC becomes 1-AUC under that reading. `cohorts.load_cohort` joins on
# the subject ID rather than on row position -- the 2026-08 data refresh changed N in
# both directions (ACT 95 -> 100, CogTE 76 -> 74), so row position is no longer a
# valid key.
COHORT = {
    "act": {
        "dir": RCT_DIR / "act",
        "id_col": "ID",
        "default_fold_col": "fold_id_MEM_bi",
        "fold_choices": ("fold_id_MEM_bi", "fold_id_EF_bi"),
        # Fold columns are stratified on one outcome; pairing them with the other
        # outcome's labels silently breaks the stratification.
        "fold_col_for_label": {"mem": "fold_id_MEM_bi", "ef": "fold_id_EF_bi"},
        "label_col_in_csv": {"mem": "MEM_decline_bi", "ef": "EF_decline_bi"},
        # The continuous change scores the binary columns were thresholded from.
        # Positive = improvement (see cohorts.Cohort.labels for the verification).
        "change_col_in_csv": {"mem": "delay_recall_ch", "ef": "exe_composite_ch"},
        # BASELINE LEVEL, not change. A level is a far better-powered target than its own
        # change score -- differencing two measurements amplifies noise while the level does
        # not -- so a representation that cannot predict the level has no chance on the
        # change. Reported as the well-powered reference the change results are read against.
        # ACT is a 2x2 FACTORIAL, not four unrelated arms: SOP (speed-of-processing
        # cognitive training) crossed with cycling, with MLA+stretch as the double control.
        #     4 = SOP + cycling (n=21)    3 = SOP only    (n=23)
        #     2 = cycling only  (n=27)    1 = MLA+stretch (n=24)
        # Analysing it as a factorial matters for power: the SOP MAIN EFFECT is 44 vs 51,
        # i.e. the whole n=95 cohort, where a pairwise "SOP only vs cycling only" contrast
        # would be 23 vs 27. `exercise_vs_no` is deliberately included as the design's
        # POSITIVE CONTROL -- aerobic-exercise effects on resting-state FC are far better
        # documented than cognitive-training effects, so if the delta cannot detect cycling
        # it cannot detect SOP either and a null on SOP says nothing.
        "arm_col": "group_assignment_T1",
        "arm_names": {1: "MLA+stretch", 2: "cycling only", 3: "SOP only",
                      4: "SOP+cycling"},
        "arm_contrasts": {
            "cogarm":  {1: [3, 4], 0: [1, 2]},      # SOP main effect,     n=95
            "exarm":   {1: [2, 4], 0: [1, 3]},      # cycling main effect, n=95
            "ctplus":  {1: [3], 0: [4]},            # SOP alone vs SOP+cycling, n=44
        },
        # Follow-up levels, the T2 counterpart of level_col_in_csv. Needed for the
        # single-timepoint cross-sectional arm: T1 features against T1 cognition and T2
        # features against T2 cognition are the same design measured twice, which is an
        # internal replication the study did not have.
        "level_t2_col_in_csv": {"mem": "calc_delay_recall_tscore_T2",
                                "ef": "executive_composite_T2"},
        "level_col_in_csv": {"mem": "calc_delay_recall_tscore_T1",
                             "ef": "executive_composite_T1"},
    },
    "cogte": {
        "dir": RCT_DIR / "cogte",
        "id_col": "ID",
        # Historical CLI used fold_id_EM_bi as an alias for fold_id_MEM_bi.
        "default_fold_col": "fold_id_MEM_bi",
        "fold_choices": ("fold_id_MEM_bi", "fold_id_EF_bi"),
        "fold_col_for_label": {"mem": "fold_id_MEM_bi", "ef": "fold_id_EF_bi"},
        "label_col_in_csv": {"mem": "MEM_decline_bi", "ef": "EF_decline_bi"},
        "change_col_in_csv": {"mem": "delay_recall_ch", "ef": "exe_composite_ch"},
        # GROUP 1 = cognitive training (n=48), GROUP 2 = MLA (n=26). Recovered from the
        # analysis table rather than assumed: its `learnergroup1learner2nonlearnermla`
        # column codes 3 = "mla" and that code falls exactly on GROUP == 2 (crosstab
        # 13/35/0 against 0/0/26). No combined arm, so no `ctplus` contrast here.
        "arm_col": "GROUP_1.0",
        "arm_names": {1: "cognitive training", 2: "MLA"},
        "arm_contrasts": {"cogarm": {1: [1], 0: [2]}},
        "level_t2_col_in_csv": {"mem": "DelayRecall_tscore_2.0",
                                "ef": "executive_composite_2.0"},
        "level_col_in_csv": {"mem": "DelayRecall_tscore_1.0",
                             "ef": "executive_composite_1.0"},
    },
    # Added with the 2026-08 refresh; outcomes assembled 2026-08-24 by
    # scripts/make_breathe_labels.py from breathe_cognition0323.csv +
    # breathe_ef_responder_4task.csv. Its TR is not documented anywhere in the package --
    # see temporal.COHORT_TR.
    #
    # Unlike ACT and CogTE, its two outcomes cover DIFFERENT subsets of the imaged
    # subjects (MEM 105/107, EF 96/107), because the source tables have different
    # follow-up coverage. Callers must therefore mask per label -- see
    # cohorts.Cohort.labelled_mask -- rather than assuming every row is scorable.
    "breathe": {
        "dir": RCT_DIR / "breathe",
        "id_col": "ID",
        "default_fold_col": "fold_id_MEM_bi",
        "fold_choices": ("fold_id_MEM_bi", "fold_id_EF_bi"),
        "fold_col_for_label": {"mem": "fold_id_MEM_bi", "ef": "fold_id_EF_bi"},
        "label_col_in_csv": {"mem": "MEM_decline_bi", "ef": "EF_decline_bi"},
        "change_col_in_csv": {"mem": "delay_recall_ch", "ef": "exe_composite_ch"},
        # Written by scripts/make_breathe_labels.py: the session-1 EM/EF composites the
        # change scores were formed from, on the session-1 scale.
        "level_t2_col_in_csv": {"mem": "em_post", "ef": "ef_post"},
        "level_col_in_csv": {"mem": "em_pre", "ef": "ef_pre"},
        # Arm 1 mixes 26 randomised participants with 26 non-randomised healthy controls;
        # `cognitive_status == 0` marks the controls and occurs in NO other arm. They are
        # younger (68.7 vs 74.4), more often female and nearly an SD higher on baseline EF,
        # so they are excluded from every analysis: they received no trial intervention, and
        # leaving them in both inflates between-subject variance on the covariates the
        # positive control reads and contaminates any change or arm contrast. Removing them
        # leaves the balanced 26 / 27 / 28 that a three-arm randomisation should show.
        # Applied by Cohort.labelled_mask, so every caller inherits it.
        "exclude_where": ("cognitive_status", 0),
        # 1 = RFB+CT (n=26), 2 = IR+CT (n=27), 3 = IR alone (n=28), after the HC
        # exclusion above. The mapping is not recorded in any source table; it was supplied
        # directly and is written here so it cannot be lost.
        #
        # `cogarm` here is the CLEANEST cognitive-training contrast in the three trials:
        # IR+CT vs IR alone holds the non-cognitive component IDENTICAL on both sides, so
        # the only difference is CT. ACT's `cogarm` is a factorial main effect averaging
        # over cycling, which is a weaker form of the same question.
        #
        # `ctplus` is NOT homogeneous with ACT's. In ACT it means "add cycling on top of
        # SOP" (SOP alone vs SOP+cycling). Here both sides already have CT and the
        # companion differs (IR vs RFB), so it is "which non-cognitive companion" instead.
        # Pool the two only with that caveat stated.
        "arm_col": "group_assignment_T1",
        "arm_names": {1: "RFB+CT", 2: "IR+CT", 3: "IR alone"},
        "arm_contrasts": {
            "cogarm": {1: [2], 0: [3]},     # IR+CT vs IR alone, n=55
            "ctplus": {1: [2], 0: [1]},     # IR+CT vs RFB+CT,   n=53
        },
    },
    # pSOPT, added 2026-09-17. A fourth independent validation cohort, and the one with the
    # cleanest imaging geometry in the set:
    #
    #   * [44, 424, 600, 2] -- 600 frames, and 43/44 T1 plus 44/44 T2 sessions are the FULL
    #     600 (one T1 at 467). At the official tlen 200, and at the 280 pinned below, this
    #     cohort CROPS and never pads, so the 2026-09-01 pad lock is exercised on 0 of 88
    #     sessions. Every padding-fraction confound in the historical input audit 4a is absent here.
    #   * TR 1.375 (user-confirmed), a value no other cohort occupies -- it sits between ACT
    #     (1.03) and CogTE (2.5). Combined with the point above, psopt is the one cohort that
    #     separates TR from padding fraction, which ACT/CogTE/BREATHE/ADNI all confound.
    #   * Its ECG window grid tiles exactly the 200-frame input window, 10 windows of 27.5 s
    #     = one per temporal patch. See meta.json.
    #
    # `labels.csv` and `subjects.txt` are BUILT by scripts/prepare_psopt.py -- psopt shipped
    # raw tables, not a prepared cohort. The mem/ef construct mapping and the `change >= 0`
    # binary rule are documented there; both were matched to the existing cohorts rather than
    # chosen, so a psopt number is comparable to an ACT/CogTE/BREATHE number.
    "psopt": {
        "dir": RCT_DIR / "psopt",
        "id_col": "ID",
        "default_fold_col": "fold_id_MEM_bi",
        "fold_choices": ("fold_id_MEM_bi", "fold_id_EF_bi"),
        "fold_col_for_label": {"mem": "fold_id_MEM_bi", "ef": "fold_id_EF_bi"},
        "label_col_in_csv": {"mem": "MEM_decline_bi", "ef": "EF_decline_bi"},
        "change_col_in_csv": {"mem": "delay_recall_ch", "ef": "exe_composite_ch"},
        "level_col_in_csv": {"mem": "delay_recall_T1", "ef": "ef_T1"},
        "level_t2_col_in_csv": {"mem": "delay_recall_T2", "ef": "ef_T2"},
        # Two arms, 23 vs 21 among the imaged subjects. `cogarm` is named to match the
        # other cohorts' cognitive-arm contrast so cross-cohort code needs no special case;
        # MLA is the same active-control label CogTE uses for its arm 2.
        "arm_col": "arm_code",
        "arm_names": {1: "psopt", 2: "MLA"},
        "arm_contrasts": {"cogarm": {1: [1], 0: [2]}},
    },
}

#: Cohorts with an outcome table, i.e. the ones the supervised entrypoints accept.
LABELLED_COHORTS = tuple(n for n in COHORT if (COHORT[n]["dir"] / "labels.csv").exists())

# Attach each cohort's file paths and its repetition time / temporal geometry. Applied
# at load time by cohorts.load_cohort and folded into the feature cache key, so no
# parallel resampled copies of data/rct/ are needed. See temporal.py for why CogTE is
# resampled and ACT is not.
for _name, _meta in COHORT.items():
    _d = _meta["dir"]
    # Two imaging releases per cohort, the same recordings under different normalisation:
    #
    #   bold_amp     <cohort>_A424_T1T2_raw_combined.pt   (2026-08-24) amplitude preserved
    #   bold_zscore  bold_T1T2.pt                         (2026-08-19) per-ROI z-scored
    #
    # Verified: per-ROI z-scoring `bold_amp` reproduces `bold_zscore` to 1.3e-04 (ACT),
    # 1.3e-05 (CogTE), 4.4e-05 (BREATHE), with identical subject order and NaN masks. The
    # z-scored release has every per-ROI sd at exactly 1.0000, i.e. between-ROI amplitude
    # is gone -- the same defect the deleted ADNI `denoise-old` release had. Since the
    # profile out-predicts frozen BrainLM on every ADNI covariate, amplitude is not a
    # nuisance here and `bold_amp` is the default. `bold_zscore` is kept as the paired
    # control: FC is identical across the two (Pearson is scale-free), so a difference
    # between them is attributable to amplitude alone.
    _cap = {"act": "ACT", "cogte": "CogTE", "breathe": "BREATHE",
            "psopt": "pSOPT"}[_name]
    _meta["bold_amp"] = _d / f"{_cap}_A424_T1T2_raw_combined.pt"
    # psopt ships its z-scored release under the same `<cap>_..._z_transformed_...` naming as
    # its raw one instead of the legacy `bold_T1T2.pt` filename the 2026-08 cohorts used.
    # Verified equivalent in kind: per-ROI z-scoring psopt's raw release reproduces the
    # shipped z file to 1.5e-03 with identical subject order and NaN masks, and every
    # per-ROI sd in it is 1.0000, i.e. between-ROI amplitude is gone -- the same relationship
    # the other three have between their two releases.
    _zs = _d / f"{_cap}_A424_T1T2_z_transformed_combined.pt"
    _meta["bold_zscore"] = _zs if _zs.exists() else _d / "bold_T1T2.pt"
    # Third release (2026-08-31): Schaefer-400 + Tian-S3 = 450 ROIs. This is the
    # parcellation Brain-JEPA requires, and it is NOT interchangeable with
    # the A424 files -- it carries a different denoising. Measured global-signal / per-ROI-sd:
    #
    #     cohort    A424        Schaefer
    #     act       0.152       0.599
    #     cogte     0.248       0.503
    #     breathe   0.152       0.620
    #
    # The A424 release had global signal suppressed; this one did not. That axis is worth
    # 0.047 AUC on ADNI CN/AD (GSR: 0.8076 -> 0.7608), so baselines must be recomputed on
    # THIS file rather than borrowed from the A424 runs.
    #
    # Row order is verified against the A424 release by per-subject valid length (a
    # volume-level censoring artefact, identical for the same subject across
    # parcellations): exact 100/100 for ACT, 107/107 for BREATHE and, since the 2026-09-01
    # refresh, 74/74 for CogTE. That refresh completed CogTE from 66 subjects to the same
    # 74 the A424 release has (adding sub-1032..sub-1040 except sub-1037's neighbour
    # sub-1036, which is absent from both), so the cohort-specific subject list it used to
    # need is gone and `subjects.txt` applies to both parcellations. `subjects_schaefer`
    # is kept as a mechanism for any future release that ships a subset.
    _meta["bold_schaefer"] = _d / f"{_cap}_Schaefer400Tian_T1T2_raw_combined.pt"
    _sch_subj = _d / f"{_cap}_Schaefer400Tian_T1T2_raw_subjects.txt"
    _meta["subjects_schaefer"] = _sch_subj if _sch_subj.exists() else None
    _meta["bold"] = _meta["bold_amp"] if _meta["bold_amp"].exists() else _meta["bold_zscore"]
    _meta["subjects"] = _d / "subjects.txt"
    _meta["labels_csv"] = _d / "labels.csv"
    _meta["native_tr"] = COHORT_TR.get(_name)
    # psopt ships the SAME recordings under two denoisings, delivered 2026-09-17 in
    # `pSOPT_visits12_noGSR/` and `pSOPT_visits12_GSR/` (labels.csv, subjects.txt, meta.json
    # and the ECG tables stay in the parent directory).
    #
    # THE DEFAULT IS noGSR, on the user's instruction 2026-09-17: psopt is to match the other
    # three, and act/cogte/breathe are noGSR. An earlier note in this file inferred the
    # opposite -- it read "the A424 release had global signal suppressed" off the MEASURED
    # global-signal/per-ROI-sd (0.152 act / 0.248 cogte / 0.152 breathe, against 0.599/0.503/
    # 0.620 for the Schaefer builds). That inference was wrong about the pipeline: none of the
    # three has GSR applied, and the low values come from the rest of their confound
    # regression. There is no GSR provenance recorded for any of them; data/README.md carries
    # none.
    #
    # THE MEASURED DISCREPANCY DOES NOT GO AWAY BY PICKING THE MATCHING LABEL, and must be
    # carried with every psopt-vs-others comparison:
    #
    #                   GS/per-ROI-sd    median per-ROI sd
    #   psopt noGSR         0.720             6.23
    #   psopt GSR           0.527             5.25
    #   act / breathe       0.152             0.73 / 0.70
    #   cogte               0.248             0.68
    #   ADNI _off           0.351             0.52
    #
    # So the label now matches while the measurement differs 4.7x on the global signal and
    # ~9x on scale. Both psopt builds are raw parcel intensity (stored median 2914), which
    # the re-export did not change. Hence `--gain_match` and `--gsr` remain necessary
    # downstream controls whenever psopt is compared with the other three; see
    # scripts/fingerprint_rct.py and scripts/mci_axis_transfer.py.
    # `*_scaled` is preferred when present: scripts/rebuild_psopt_scaled.py divides each
    # session by a scalar so its median per-ROI sd is exactly 0.700, the median of the three
    # delivered RCT releases (0.733 / 0.679 / 0.703), removing the ~9x amplitude mismatch at
    # source so `--gain_match` is no longer needed for psopt. What that rebuild does NOT and
    # cannot fix is the GLOBAL SIGNAL (0.720 against 0.152-0.248): a scalar cannot change the
    # ratio of a shared across-ROI component to the per-ROI spread, so `--gsr` remains
    # necessary and the flag stays. The upstream spec for fixing it is in that script's
    # docstring. The `_z_transformed_` variants are copied unchanged, since every per-ROI sd
    # in them is already exactly 1.000 and a scalar is meaningless there.
    _gsr, _nogsr = _d / "pSOPT_visits12_GSR", _d / "pSOPT_visits12_noGSR"
    if (_d / "pSOPT_visits12_noGSR_scaled").is_dir():
        _gsr, _nogsr = _d / "pSOPT_visits12_GSR_scaled", _d / "pSOPT_visits12_noGSR_scaled"
    if _gsr.is_dir() and _nogsr.is_dir():
        for _k, _stem in (("bold_amp", "A424_T1T2_raw"),
                          ("bold_zscore", "A424_T1T2_z_transformed"),
                          ("bold_schaefer", "Schaefer400Tian_T1T2_raw")):
            _meta[_k] = _nogsr / f"{_cap}_{_stem}_combined.pt"
            _meta[_k + "_gsr"] = _gsr / f"{_cap}_{_stem}_combined.pt"
        # `bold` was set earlier in this loop from the pre-override paths, which for psopt
        # no longer exist, so it is repointed at the default release here.
        _meta["bold"] = _meta["bold_amp"]
    _meta.update(COHORT_RESAMPLE.get(_name, {"target_tr": None, "tlen": None}))


def temporal_spec(cohort: str) -> dict:
    """(native_tr, target_tr, tlen) for a cohort, as keyword arguments.

    `native_tr` here is the code-table value; `cohorts.cohort_tr` additionally honours a
    `meta.json` in the cohort directory and is what the entrypoints use.
    """
    m = COHORT[cohort]
    return {"native_tr": m["native_tr"], "target_tr": m["target_tr"], "tlen": m["tlen"]}


def require_build(tag: str, kind: str = "adni") -> Path:
    """Path to a derived build, or a clear error naming what exists instead.

    Entry points default to a tag, and a default that silently points at a deleted build
    surfaces as a FileNotFoundError deep inside a job that has already spent GPU time. Worse,
    when several builds differ only by an input-recipe suffix (`_off` official, `_offc`
    official plus the dataset constant, `_amp` the superseded per-scan-scalar variant, bare
    for per-ROI z-scored), a wrong suffix is not obviously wrong from the traceback -- and
    that is exactly the class of mistake that cost this project a full round of results.
    """
    root = DERIVED_DIR / ("adni_paired" if kind == "paired" else "adni")
    p = root / f"ADNI_x_{tag}.pt"
    if p.exists():
        return p
    have = sorted(q.name[len("ADNI_x_"):-3] for q in root.glob("ADNI_x_*.pt"))
    raise FileNotFoundError(
        f"derived build {tag!r} not found under {root}. Available: {have}. "
        "Suffixes: _off = BrainLM's official recipe (per-ROI mean-centring, no division), "
        "_offc = official plus the dataset constant, no suffix = per-ROI z-scored control. "
        "The _amp variants were deleted on 2026-08-25: they applied a per-scan scalar "
        "division that is not part of the upstream pipeline.")
