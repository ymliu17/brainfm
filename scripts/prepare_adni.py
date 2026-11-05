#!/usr/bin/env python3
"""Build the ADNI stage-1 tensors ([N,424,T] + [N]) that finetune_adni.py expects.

RECONSTRUCTION: the package ships per-session `.pt` files + cognitive CSVs but no script
that assembles them, and the manuscript does not state the session/label/crop choices.
Every such choice is a CLI flag here and the resolved values are printed, so a reader can
see exactly what was assumed.

Scan release
------------
`--scan_dir` selects which ADNI preprocessing release to build from; the release
directory name goes into the output tag and into the manifest, so builds from different
releases cannot collide on a filename. The current one is `data/adni/raw`
(1617 sessions / 605 subjects, 2026-08-19). Its session stems carry an `_ADNI2` / `_ADNI3`
suffix that the clinical `FILE_NAME` column does not, so the suffix is stripped before
matching.

Label source
------------
`ADNI_UMNPSYCH_COG_forXL.csv` carries FILE_NAME but no diagnosis;
`ADNI_UMNPSYCH_COG_forXL_new.csv` carries DX/DX_bl but no FILE_NAME. The two files
have identical row counts and their subject IDs match row-for-row (verified: 1.0
agreement on ID and on ADNI_MEM), so FILE_NAME is transferred positionally.

  --label_col DX     diagnosis at the scan visit: Dementia=1 / CN=0   (default)
  --label_col DX_bl  baseline diagnosis:          AD=1       / CN=0

Session handling
----------------
Subjects contribute several sessions. `--sessions all` keeps them, which lets a
random split place the same subject on both sides; `--sessions first` (default)
keeps one session per subject so any downstream split is leakage-free.

Example:
  python scripts/prepare_adni.py --out_dir data/derived/adni
"""
from __future__ import annotations

import argparse
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from temporal import resample_to_length  # noqa: E402
from paths import ADNI_COG_CSV, ADNI_COG_NEW_CSV, ADNI_SCAN_DIR, DERIVED_DIR  # noqa: E402

LABELS = {"DX": ("Dementia", "CN"), "DX_bl": ("AD", "CN")}

#: Ordinal severity targets. The binary maps above discard MCI entirely -- 686 of the 1742
#: usable scans (279 of 740 subjects at first session), i.e. most of the cohort. Two reasons
#: that matters beyond sample size:
#:
#:  1. a boundary fitted on the CN / Dementia extremes is an EXTRAPOLATION in the middle of
#:     the axis, and the RCT participants are cognitively normal-to-mildly-impaired older
#:     adults who all sit in that middle. An axis meant to detect movement there has to be
#:     resolved there.
#:  2. an ordinal target forces the representation to encode severity rather than only
#:     separate the endpoints.
#:
#: Values are normalised to [0, 1] so the same regression head and metric work for binary
#: and ordinal targets.
#: Continuous cross-sectional cognitive LEVEL targets. These are the most reliable targets
#: ADNI offers and the ones whose construct matches the RCT outcome:
#:
#:   visit-to-visit reliability of the LEVEL   ADNI_MEM r=0.924   ADNI_EF r=0.888
#:   consecutive-change correlation            ADNI_MEM r=-0.292  ADNI_EF r=-0.343
#:
#: so the level is a stable trait-like quantity while the two-timepoint change is
#: measurement-error dominated (a negative autocorrelation of successive differences is the
#: algebraic signature of that). The RCT outcome is a CHANGE in the same composites, but the
#: axis it is measured against does not have to be a change axis -- and cannot usefully be
#: one here, because the ADNI change target carries no learnable signal (nothing predicts
#: it, including baseline cognition; see the original cohort audit).
#:
#: A memory-ability axis is also arguably better matched to the intervention hypothesis than
#: a diagnosis axis: CN-vs-AD separation is driven partly by things six months of exercise or
#: cognitive training cannot move, whereas the composite is the quantity the trial measures.
CONTINUOUS = {
    "ADNI_MEM": "memory composite level (matches the RCT memory outcome's construct)",
    "ADNI_EF": "executive composite level (matches the RCT executive outcome's construct)",
}

ORDINAL = {
    "DX_ord": {"CN": 0.0, "MCI": 0.5, "Dementia": 1.0},
    "DX_bl_ord": {"CN": 0.0, "SMC": 0.25, "EMCI": 0.5, "LMCI": 0.75, "AD": 1.0},
}

#: Trailing tags on the session stems that the clinical FILE_NAME column does not carry.
#: The 2026-08-24 release ends in `_ADNI2` / `_ADNI3`; the 2026-08-25 fMRIPrep release adds
#: a pipeline/batch suffix on top of that (`_fmriprep`, or `_addition187` for the subjects
#: that were missing from the first export and were re-run).
COHORT_SUFFIX = re.compile(r"_ADNI[23](?:_(?:fmriprep|addition187))?$")

#: Which export batch a scan came from, or None for releases that do not label batches.
#: Recorded in the manifest because the two batches of the fMRIPrep release differ in
#: measured ways -- `addition187` is entirely ADNI3, its median per-scan gain is ~9x lower
#: within ADNI3, and its diagnosis mix differs from the main batch (5.9% vs 14.1% Dementia,
#: chi2 p<1e-4). Whether the fMRIPrep version and settings were identical is not documented,
#: so batch has to travel with the data as a candidate technical covariate.
BATCH_SUFFIX = re.compile(r"_ADNI[23]_(fmriprep|addition187)$")


def session_stem(path: Path) -> str:
    """Session key that matches the clinical table's FILE_NAME column."""
    return COHORT_SUFFIX.sub("", path.stem)


def scan_batch(path: Path) -> str | None:
    """Export batch of a scan, or None if the release does not label them."""
    m = BATCH_SUFFIX.search(path.stem)
    return m.group(1) if m else None


def scan_protocol(path: Path) -> str | None:
    """ADNI2 / ADNI3 from the stem, which is more reliable than inferring it from length."""
    m = re.search(r"_(ADNI[23])(?:_(?:fmriprep|addition187))?$", path.stem)
    return m.group(1) if m else None


def build_manifest(scan_dir: Path, label_col: str) -> pd.DataFrame:
    """Sessions with imaging and a usable label. `label` is 1/0 for the binary maps and a
    graded value in [0, 1] for the ORDINAL ones."""
    with_files = pd.read_csv(ADNI_COG_CSV, low_memory=False)
    with_dx = pd.read_csv(ADNI_COG_NEW_CSV, low_memory=False)
    if len(with_files) != len(with_dx):
        raise SystemExit("the two cognitive CSVs no longer align row-for-row")
    aligned = with_files["ID"].str.replace("sub-ADNI", "", regex=False).values == with_dx[
        "PTID"
    ].str.replace("_", "", regex=False).values
    if not aligned.all():
        raise SystemExit(f"row-wise ID mismatch in {(~aligned).sum()} rows; refusing to guess")
    with_dx = with_dx.assign(FILE_NAME=with_files["FILE_NAME"].values)

    scans = sorted(scan_dir.glob("*.pt"))
    if not scans:
        raise SystemExit(f"no .pt scans under {scan_dir}")
    available = {session_stem(p): p for p in scans}
    if len(available) != len(scans):
        raise SystemExit(f"{scan_dir}: session stems collide after stripping the cohort suffix")
    matched = with_files["FILE_NAME"].isin(available).sum()
    print(f"{len(scans)} scans in {scan_dir.name}; {matched} have a clinical row")
    if label_col in CONTINUOUS:
        df = (
            with_dx[with_dx["FILE_NAME"].isin(available) & with_dx[label_col].notna()]
            .drop_duplicates("FILE_NAME")
            .sort_values("FILE_NAME")
            .reset_index(drop=True)
        )
        df["path"] = df["FILE_NAME"].map(available)
        df["label"] = pd.to_numeric(df[label_col], errors="coerce").astype(float)
        df = df[np.isfinite(df["label"])].reset_index(drop=True)
        df["dx_stage"] = df["DX"] if "DX" in df.columns else None
        print(f"continuous target {label_col}: n={len(df)} "
              f"mean {df['label'].mean():+.3f} sd {df['label'].std():.3f}")
        return df
    if label_col in ORDINAL:
        levels = ORDINAL[label_col]
        src = label_col[:-4] if label_col.endswith("_ord") else label_col
        df = (
            with_dx[with_dx["FILE_NAME"].isin(available) & with_dx[src].isin(levels)]
            .drop_duplicates("FILE_NAME")
            .sort_values("FILE_NAME")
            .reset_index(drop=True)
        )
        df["path"] = df["FILE_NAME"].map(available)
        df["label"] = df[src].map(levels).astype(float)
        df["dx_stage"] = df[src]
        print("ordinal level counts:", df["dx_stage"].value_counts().to_dict())
        return df
    pos, neg = LABELS[label_col]
    df = (
        with_dx[with_dx["FILE_NAME"].isin(available) & with_dx[label_col].isin([pos, neg])]
        .drop_duplicates("FILE_NAME")
        .sort_values("FILE_NAME")
        .reset_index(drop=True)
    )
    df["path"] = df["FILE_NAME"].map(available)
    df["label"] = (df[label_col] == pos).astype(int)
    df["dx_stage"] = df[label_col]
    return df


# Scan length is an acquisition-protocol fingerprint, not a free parameter:
#   140      -> ADNI2 / ADNIGO, TR 3.0 s
#   197-200  -> mostly ADNI3 basic, TR 3.0 s
#   >=750    -> ADNI3 advanced, TR 0.607 s
# The three groups differ in AD prevalence (0.32 / 0.17 / 0.20 in the one-session
# set), so any evaluation should be reported per group as well as overall.
def protocol_group(n_timepoints: int) -> str:
    """Acquisition-protocol group. Length thresholds account for fMRIPrep dropping
    non-steady-state volumes, which moved 140 -> 134, 197 -> 191 and 200 -> 194."""
    if n_timepoints >= 600:
        return "advanced_TR0.6"
    if n_timepoints <= 145:
        return "adni2_TR3.0"
    if n_timepoints <= 250:
        return "basic_TR3.0"
    return "unknown"


def native_tr(n_timepoints: int) -> float:
    """Repetition time in seconds, inferred from the run length.

    Both acquisitions last ~10 minutes, so the length separates them: the basic protocol
    (ADNI2/ADNIGO and ADNI3 basic) is TR 3.0 s and runs 134-200 volumes, and ADNI3 advanced
    is multiband at TR 0.607 s and runs 744-976 volumes.

    The old `>= 750` threshold silently misread the fMRIPrep release, which contains runs of
    744 and 970-975 volumes -- fMRIPrep drops non-steady-state volumes, so every length
    shifted down and 744 fell on the wrong side, which would have labelled a 7.5-minute
    multiband run as TR 3.0 (37 minutes). The band between the two groups is now explicit
    and anything inside it raises rather than being guessed.
    """
    if n_timepoints <= 250:
        return 3.0
    if n_timepoints >= 600:
        return 0.607
    raise ValueError(
        f"cannot infer TR from {n_timepoints} volumes: it falls between the basic protocol "
        f"(<=250 volumes at TR 3.0 s) and ADNI3 advanced (>=600 at TR 0.607 s). Supply the "
        f"TR explicitly rather than letting it be guessed.")


def resample_to_tr(sig: torch.Tensor, tr_from: float, tr_to: float) -> torch.Tensor:
    """Resample a [R, T] run from tr_from to tr_to with polyphase FIR filtering.

    Why this matters: BrainLM was pretrained on UK Biobank alone (TR 0.735 s), so one
    20-frame token spanned ~14.7 s during pretraining. HCP is a downstream evaluation
    set in that paper, not part of the corpus. At ADNI's
    TR 3.0 s the same token spans 60 s -- a 4x mismatch in what a token means. Putting
    every run on a common grid near the pretraining TR makes token duration comparable.

    Caveat that must travel with any result from this: TR 3.0 s samples at a Nyquist
    frequency of 0.167 Hz, TR 0.735 s at 0.68 Hz. Resampling upward cannot recreate the
    0.167-0.68 Hz band that was never acquired. Token *duration* is harmonised; spectral
    content is not. Expect partial recovery at best.
    """
    if abs(tr_from - tr_to) < 1e-9:
        return sig
    ratio = Fraction(tr_from / tr_to).limit_denominator(200)
    out = signal.resample_poly(sig.numpy(), ratio.numerator, ratio.denominator, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(out)).float()


def main():
    ap = argparse.ArgumentParser(description="Assemble ADNI stage-1 tensors")
    ap.add_argument("--scan_dir", default=str(ADNI_SCAN_DIR))
    ap.add_argument("--label_col", default="DX",
                    choices=sorted(LABELS) + sorted(ORDINAL) + sorted(CONTINUOUS),
                    help="DX / DX_bl are binary (MCI dropped); DX_ord / DX_bl_ord are graded "
                         "severity in [0,1] and keep MCI -- see ORDINAL above")
    ap.add_argument("--dx_filter", nargs="+", default=None,
                    choices=("CN", "MCI", "Dementia", "EMCI", "LMCI", "SMC", "AD"),
                    help="restrict the fitting population to these diagnostic stages. "
                         "`--dx_filter MCI` with a CONTINUOUS label is the RCT-matched "
                         "build: 686 sessions from 299 subjects, all with ADNI_MEM and "
                         "ADNI_EF, and the largest within-group cognitive spread of the "
                         "three stages. Values are matched against `dx_stage`, which is "
                         "DX for continuous targets and the label column otherwise.")
    ap.add_argument("--sessions", default="first", choices=("first", "all"))
    ap.add_argument("--n_roi", type=int, default=None,
                    help="expected ROI count per scan; defaults to config.NUM_BRAIN_VOXELS "
                         "(424, the A424 atlas). Set 450 to build the Schaefer-400 + "
                         "Tian-S3 release that Brain-JEPA requires. The check is kept "
                         "rather than inferred so a truncated or mis-parcellated scan is "
                         "a hard error instead of a silently wrong tensor.")
    ap.add_argument("--tlen", type=int, default=140, help="crop length in frames; multiple of 20")
    ap.add_argument("--short_policy", default="off",
                    choices=("off", "adaptive", "stretch"),
                    help="what to do with a run that cannot supply --tlen. `off` is the "
                         "historical behaviour: drop it (or, with --target_tr, resample "
                         "EVERY run and then drop whatever is still short). `adaptive` "
                         "decides per run -- crop if it already has --tlen, zero-pad if it "
                         "is short by at most --pad_tolerance frames, resample to "
                         "--target_tr only if it is short by more than that. This exists "
                         "because ADNI's shortfalls are bimodal: 453 runs are 197 frames "
                         "and 280 are 200, the same protocol at the same TR, while 250 are "
                         "140. The 197s are padded -- stretching them would filter every "
                         "acquired sample to fix 1.5%% of the length. The 140s are "
                         "stretched to exactly --tlen -- padding them would spend 3 of 10 "
                         "tokens on zeros identical across ROIs and subjects. `stretch` is the one actually "
                         "used: crop if the run is long enough, otherwise resample it to "
                         "EXACTLY --tlen, using the whole recording and filling every "
                         "token with real signal. It needs no tolerance threshold, which "
                         "is its main advantage over `adaptive` -- 197->200 and 140->200 "
                         "stop being different kinds of operation and become the same one "
                         "at 1.015x and 1.43x. See temporal.resample_to_length.")
    ap.add_argument("--fill", default="pad",
                    choices=("auto", "pad", "stretch", "tile", "reflect"),
                    help="LOCKED 2026-09-01: a scan short of --tlen is ZERO-PADDED, after per-ROI mean-centring. Not stretched, not tiled. Decided by measurement across four comparisons -- pad beat stretch 24/25 and 26/33 cells, beat tile 19/25 on ADNI and 5/9 on CogTE, beat reflect 23/25 and 9/9 -- covering shortfalls from 30%% (ADNI 140->200) to 53%% (CogTE 95->200). Stretching filters every acquired sample; tiling injects a periodicity the recording never had, which costs the ENCODERS specifically (on CogTE all six FC/amp cells moved under 0.01 while BrainLM age fell 0.134). The other choices remain available for re-testing a genuinely new regime; they are not the default and should not be made one without numbers. "
                         "Values other than the default: `auto` restores the old tolerance "
                         "rule (pad small shortfalls, stretch large ones). "
                         "`auto` keeps the tolerance rule. The alternatives fill every "
                         "shortfall the same way: `pad` inserts frames that are identical "
                         "across every ROI and subject; `stretch` passes every acquired "
                         "sample through a polyphase filter; `tile` repeats the recording "
                         "head-to-tail, keeping all real values but leaving a "
                         "discontinuity at the seam; `reflect` tiles with alternate copies "
                         "time-reversed, so the seams are continuous. `stretch` already "
                         "lost to `pad` by 26 of 33 cells; tile and reflect are the "
                         "untested options that neither filter nor fabricate constants.")
    ap.add_argument("--max_stretch", type=float, default=3.0,
                    help="with --short_policy stretch, drop a run that would have to be "
                         "stretched by more than this. A 45-volume scan reaching 200 is a "
                         "4.4x interpolation of an aborted acquisition, not a short "
                         "recording; 140->200 is 1.43x and 100->200 is 2.0x.")
    ap.add_argument("--pad_tolerance", type=int, default=None,
                    help="with --short_policy adaptive, the largest shortfall that is "
                         "zero-padded instead of resampled. Defaults to one temporal patch "
                         "(20 frames), so at most the FINAL token is partly padding and "
                         "every earlier token is entirely real. Padding happens after "
                         "per-ROI mean-centring, so a padded frame sits exactly at that "
                         "ROI's mean -- the least informative filler available -- and the "
                         "amplitude statistics are computed on the real frames only.")
    ap.add_argument("--target_tr", type=float, default=None,
                    help="resample every run to this TR (s) before cropping. 0.735 matches "
                         "BrainLM pretraining (UK Biobank, 0.735 s), making one 20-frame "
                         "token span ~14.7s for every scan instead of 60s vs 12.1s.")
    ap.add_argument("--amplitude", default="zscore",
                    choices=("zscore", "preserve", "official", "official_offset"),
                    help="zscore: divide every ROI by its own sd (the historical path; sets "
                         "every per-ROI sd to 1.0 and destroys between-ROI amplitude). "
                         "preserve: mean-centre per ROI and divide the WHOLE scan by ONE "
                         "scalar (its median per-ROI sd), which removes the arbitrary BOLD "
                         "gain but keeps the between-ROI amplitude ratios that BrainLM was "
                         "pretrained on. Requires an amplitude-bearing release such as "
                         "data/adni/raw; on an already-z-scored release it is a no-op that "
                         "silently reproduces --amplitude zscore, so the release is checked. "
                         "official: BrainLM's OWN recipe -- per-ROI mean-centring and nothing "
                         "else, no per-scan division at all, which is what the released weights "
                         "were fitted on (src/brainlm_input.py transcribes the upstream code). "
                         "The difference from preserve is measurable: preserve puts every "
                         "scan's median per-ROI sd at exactly 1.000, so it keeps between-ROI "
                         "ratios but removes BETWEEN-SUBJECT amplitude, which upstream keeps. "
                         "The cost of official is that without per-scan normalisation the 4%% of "
                         "scans sitting 2-3 orders of magnitude off the cohort median dominate "
                         "an MSE objective, so --max_scale_dev must exclude them rather than "
                         "rescale them. Upstream never faced this: UK Biobank is single-site. "
                         "official_offset: official PLUS upstream's step 2, the per-ROI "
                         "constant median/IQR, fitted on THIS dataset by the upstream formula "
                         "(the UK Biobank constants BrainLM used are not published). The two "
                         "official modes differ by a per-ROI constant vector, so their "
                         "variance structure is identical -- FC and any mean-centred linear "
                         "probe cannot tell them apart, but the encoder is not "
                         "translation-invariant, so for the FM this isolates what a pure "
                         "shift is worth.")
    ap.add_argument("--max_scale_dev", type=float, default=3.0,
                    help="with --amplitude preserve, drop scans whose median per-ROI sd is "
                         "more than this many log2 units from the cohort median. 78/1877 "
                         "scans in data/adni/raw sit 2-3 orders of magnitude off; their "
                         "amplitude profile correlates with the same subject's other scan "
                         "at r=+0.34, BELOW the +0.44 between-subject baseline (clean "
                         "within-subject is +0.59), so they are corrupt rather than merely "
                         "rescaled and no global factor recovers them.")
    ap.add_argument("--exclude_protocol_group", nargs="*", default=[],
                    help="drop scans in these protocol_group values (see protocol_group()). "
                         "The reason this exists: BrainLM was pretrained at TR 0.72-0.735 s "
                         "and ADNI3 advanced is the only ADNI arm anywhere near it, at "
                         "TR 0.607 s, while the rest is TR 3.0 s. Resampling everything to "
                         "0.735 s was tried first and failed decisively -- every BrainLM "
                         "readout lost 0.029-0.081 AUC AND so did FC edges (-0.046). FC is "
                         "computed from the timeseries without any token structure, so a "
                         "loss there cannot be a pretrain-distribution effect; 4x upsampling "
                         "was damaging the data. Excluding the minority arm and keeping "
                         "native TR is the remaining way to make the set TR-homogeneous.")
    ap.add_argument("--protocol", default=None, choices=("ADNI2", "ADNI3"),
                    help="restrict to one acquisition protocol, read from the filename "
                         "rather than guessed from the run length")
    ap.add_argument("--roi_qc_policy", choices=("legacy", "sparse"), default="legacy")
    ap.add_argument("--max_dead_roi", type=int, default=0,
                    help="drop scans with more than this many zero-variance parcels "
                         "(default 0: any dead parcel disqualifies the scan)")
    ap.add_argument("--out_dir", default=str(DERIVED_DIR / "adni"))
    args = ap.parse_args()
    n_roi = args.n_roi or C.NUM_BRAIN_VOXELS

    if args.tlen % C.TIMEPOINT_PATCHING_SIZE:
        raise SystemExit(f"--tlen must be a multiple of {C.TIMEPOINT_PATCHING_SIZE}")

    scan_dir = Path(args.scan_dir)
    df = build_manifest(scan_dir, args.label_col)
    if args.dx_filter:
        # Restrict the fitting population to a diagnostic stage. The whole reason the
        # ORDINAL note above exists: an axis fitted on the CN/Dementia extremes is an
        # extrapolation in the middle, and every RCT participant sits in the middle -- ACT
        # and CogTE are MCI throughout, BREATHE is mixed CN/MCI. Fitting a CONTINUOUS
        # cognitive target inside MCI alone makes both the population and the construct
        # match the RCT side, and MCI is not a narrow band: its within-group spread on
        # ADNI_MEM is 0.840 against 0.742 for CN and 0.825 for Dementia, i.e. the LARGEST
        # of the three, so there is more usable variance here than in either endpoint
        # group.
        keep = df["dx_stage"].isin(args.dx_filter)
        print(f"--dx_filter {'+'.join(args.dx_filter)}: keeping {int(keep.sum())}/{len(df)} "
              f"sessions; dropped stages "
              f"{df.loc[~keep, 'dx_stage'].value_counts().to_dict()}")
        df = df[keep].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"--dx_filter {args.dx_filter} left no sessions")
        if args.label_col in CONTINUOUS:
            print(f"  target {args.label_col} within {'+'.join(args.dx_filter)}: "
                  f"n={len(df)} mean {df['label'].mean():+.3f} sd {df['label'].std():.3f}")
    df["batch"] = [scan_batch(Path(p)) for p in df["path"]]
    df["scan_protocol"] = [scan_protocol(Path(p)) for p in df["path"]]
    if args.protocol:
        keep_p = df["scan_protocol"] == args.protocol
        print(f"--protocol {args.protocol}: keeping {int(keep_p.sum())}/{len(df)} sessions "
              f"(the fMRIPrep release's ADNI2 and ADNI3 arms differ in intensity scale by "
              f"~1000x, so they must not be pooled without that being resolved)")
        df = df[keep_p].reset_index(drop=True)
    if df["batch"].notna().any():
        print(f"export batches: {df['batch'].value_counts(dropna=False).to_dict()}")
    print(f"matched sessions: {len(df)} over {df['PTID'].nunique()} subjects")
    if args.sessions == "first":
        df = df.drop_duplicates("PTID").reset_index(drop=True)
        print(f"keeping earliest session per subject -> {len(df)}")

    pad_tol = (args.pad_tolerance if args.pad_tolerance is not None
               else C.TIMEPOINT_PATCHING_SIZE)
    stretched = []

    keep, dropped, dead, ambiguous_tr, excluded_group = [], [], [], [], []
    padded, resampled_runs, tiled = [], [], []
    for row in df.itertuples():
        sig = torch.load(row.path, map_location="cpu", weights_only=False).float()
        if sig.shape[0] != n_roi:
            raise SystemExit(f"{row.FILE_NAME}: expected {n_roi} ROIs, got {sig.shape[0]}")
        # How short is too short. Under `adaptive` a run is only rejected if resampling
        # cannot rescue it either, so the gate is deferred to after the TR is known --
        # which also fixes an ordering bug: native_tr() was called HERE, outside the
        # try/except below that exists precisely to skip runs whose TR is ambiguous, so a
        # single 386-volume scan aborted an entire 1700-scan build.
        if args.short_policy in ("stretch", "adaptive"):
            if sig.shape[1] * args.max_stretch < args.tlen:
                dropped.append((row.FILE_NAME, int(sig.shape[1])))
                continue
        elif args.short_policy == "off":
            if not args.target_tr:
                min_raw = args.tlen
            else:
                try:
                    min_raw = int(np.ceil(args.tlen * args.target_tr / native_tr(sig.shape[1])))
                except ValueError:
                    ambiguous_tr.append((row.FILE_NAME, int(sig.shape[1])))
                    continue
            if sig.shape[1] < min_raw:
                dropped.append((row.FILE_NAME, int(sig.shape[1])))
                continue
        # QC gate: a parcel with zero variance carries no signal. Six scans in the pack
        # have dead ROIs, one with 323 of 424 dead (scripts/qc_adni.py, the original cohort audit
        # 4b-8); every experiment before that QC silently included them.
        n_dead = int((sig.std(dim=1) < 1e-6).sum())
        if args.roi_qc_policy == "sparse":
            from roi_signal_qc import require_usable
            try:
                require_usable(sig, args.tlen)
            except ValueError:
                dead.append((row.FILE_NAME, n_dead))
                continue
        elif n_dead > args.max_dead_roi:
            dead.append((row.FILE_NAME, n_dead))
            continue
        raw_t = int(sig.shape[1])
        if protocol_group(raw_t) in args.exclude_protocol_group:
            excluded_group.append((row.FILE_NAME, raw_t, protocol_group(raw_t)))
            continue
        # A run length that falls between the two acquisition protocols cannot have its TR
        # inferred. Refusing to guess is right, but one such scan must not abort a build of
        # 1700: skip it and say so. (Measured: exactly one 386-volume scan in this release.)
        try:
            native_tr(raw_t)
        except ValueError:
            ambiguous_tr.append((row.FILE_NAME, raw_t))
            continue
        # Amplitude spread measured on the FULL run. After cropping, a run that was
        # z-scored over its full length no longer has exactly-equal per-ROI sds
        # (measured p5 0.87 / p95 1.11 at tlen 140), so a post-crop test cannot tell an
        # amplitude-bearing release from a z-scored one. Pre-crop it is exact.
        sd_full = sig.std(1, unbiased=False)
        spread_full = float(sd_full.max() / sd_full.median().clamp_min(1e-12))
        tr_src = native_tr(raw_t)
        n_pad, resampled = 0, False
        if args.short_policy == "stretch":
            if raw_t < args.tlen:
                sig = resample_to_length(sig, args.tlen, dim=1)
                stretched.append((row.FILE_NAME, raw_t, tr_src * raw_t / args.tlen))
        elif args.short_policy == "adaptive":
            if raw_t >= args.tlen:
                pass                                  # long enough on its own grid
            elif args.fill in ("tile", "reflect"):
                reps = -(-args.tlen // raw_t)
                sig = torch.cat(
                    [sig if (args.fill == "tile" or k % 2 == 0) else torch.flip(sig, dims=[1])
                     for k in range(reps)], dim=1)[:, :args.tlen]
                tiled.append((row.FILE_NAME, raw_t, reps))
            elif args.fill == "pad" or args.tlen - raw_t <= pad_tol:
                # A sliver. Stretching here would run a 1.015x interpolation over all 197
                # real samples to gain 3 frames -- every acquired value gets filtered to
                # fix 1.5% of the length. Padding leaves all 197 exactly as acquired and
                # spends the shortfall on the tail of the final token.
                n_pad = args.tlen - raw_t
            else:
                # Too far to pad: 140 -> 200 would be 3 of 10 tokens made of zeros that
                # are identical for every ROI and every subject. Stretch instead, which
                # uses the whole recording and leaves every token real; the price is that
                # a token spans 42 s rather than the pretrained 14.7 s, and `trhom`
                # measured that exact axis at -0.0050 +/- 0.0151 over 80 paired cells.
                sig = resample_to_length(sig, args.tlen, dim=1)
                stretched.append((row.FILE_NAME, raw_t, tr_src * raw_t / args.tlen))
        elif args.target_tr:
            sig = resample_to_tr(sig, tr_src, args.target_tr)
            resampled = True
            if sig.shape[1] < args.tlen:
                dropped.append((row.FILE_NAME, int(sig.shape[1])))
                continue
        sig = sig[:, : args.tlen]
        sig = sig - sig.mean(1, keepdim=True)
        scale = float(sig.std(1, unbiased=False).median())
        if args.amplitude == "zscore":
            # Source scans are per-ROI z-scored over the full run (matching how the
            # ACT/CogTE tensors are stored); cropping breaks that, so re-standardize.
            sig = sig / sig.std(1, keepdim=True).clamp_min(1e-6)
        if n_pad:
            # AFTER mean-centring and after `scale` was read, so the padded frames sit at
            # each ROI's own mean and neither the amplitude statistic nor the z-scoring
            # denominator sees them.
            sig = torch.cat([sig, sig.new_zeros(sig.shape[0], n_pad)], dim=1)
            padded.append((row.FILE_NAME, raw_t, n_pad))
        if resampled:
            resampled_runs.append((row.FILE_NAME, raw_t, tr_src))
        # --amplitude preserve defers its (single, global) division to after the loop,
        # because the off-scale gate needs the cohort-wide median first.
        keep.append((sig, row, raw_t, scale, spread_full))

    if dropped:
        print(f"dropped {len(dropped)} scans shorter than {args.tlen}: {dropped}")
    if args.short_policy == "stretch":
        from collections import Counter
        print(f"--short_policy stretch (tlen {args.tlen}):")
        print(f"  native crop {len(keep) - len(stretched):5d}")
        print(f"  stretched   {len(stretched):5d}  "
              f"{dict(Counter(f'{t}f->{args.tlen}f (x{args.tlen / t:.3f}, eff TR {e:.3f}s)' for _, t, e in stretched))}")
    if args.short_policy == "adaptive":
        from collections import Counter
        print(f"--short_policy adaptive (tlen {args.tlen}, pad tolerance {pad_tol}):")
        print(f"  native crop  {len(keep) - len(padded) - len(stretched) - len(tiled):5d}")
        print(f"  zero-padded  {len(padded):5d}  "
              f"{dict(Counter(f'{t}->+{n}' for _, t, n in padded))}")
        if tiled:
            print(f"  {args.fill:<12} {len(tiled):5d}  "
                  f"{dict(Counter(f'{t}f x{r}' for _, t, r in tiled))}")
        print(f"  stretched    {len(stretched):5d}  "
              f"{dict(Counter(f'{t}f->{args.tlen}f (x{args.tlen / t:.3f}, eff TR {e:.3f}s)' for _, t, e in stretched))}")
    if dead:
        print(f"dropped {len(dead)} scans by ROI QC policy {args.roi_qc_policy} (legacy max={args.max_dead_roi}): {dead}")
    if ambiguous_tr:
        print(f"dropped {len(ambiguous_tr)} scans whose run length falls between the two "
              f"acquisition protocols, so their TR cannot be inferred: {ambiguous_tr}")
    if args.exclude_protocol_group:
        from collections import Counter
        print(f"--exclude_protocol_group {args.exclude_protocol_group}: dropped "
              f"{len(excluded_group)} scans "
              f"{dict(Counter(g for _, _, g in excluded_group))}")
        if not excluded_group:
            raise SystemExit(
                f"--exclude_protocol_group {args.exclude_protocol_group} matched nothing. "
                f"Either the group name is misspelled or this release has no such scans; "
                f"silently building an unfiltered set under a filtered tag would make the "
                f"two arms indistinguishable on disk.")

    # The off-scale gate is a QC criterion, not part of the amplitude choice, so it runs
    # in both modes: the two builds must differ ONLY in the amplitude treatment or the
    # comparison between them confounds amplitude with subject composition. On an
    # already-z-scored release every scale is ~1 and the gate is a no-op.
    scales = np.array([k[3] for k in keep])
    ref = float(np.median(scales))
    off = np.abs(np.log2(np.maximum(scales, 1e-12) / ref)) > args.max_scale_dev
    if off.any():
        worst = sorted(((float(scales[i]), keep[i][1].FILE_NAME) for i in np.where(off)[0]),
                       key=lambda t: -t[0])
        print(f"dropped {int(off.sum())} off-scale scans (cohort median per-ROI sd {ref:.4f}, "
              f"gate +/-{args.max_scale_dev:g} log2): {worst[:6]}"
              + (" ..." if len(worst) > 6 else ""))
    keep = [k for k, bad in zip(keep, off) if not bad]

    out_dir_for_offset = Path(args.out_dir)
    out_dir_for_offset.mkdir(parents=True, exist_ok=True)
    if args.amplitude in ("preserve", "official", "official_offset"):
        # Guard against silently "preserving" amplitude that the release already removed.
        # An amplitude-bearing scan has per-ROI sds that differ from each other; a
        # z-scored one has them all equal, so the ratio max/min collapses to 1.
        spread = max(k[4] for k in keep)
        if spread < 1.01:
            raise SystemExit(
                f"--amplitude preserve on release '{scan_dir.name}': on the full runs the "
                f"largest per-ROI sd is only {(spread - 1) * 100:.3f}% above the median, i.e. "
                "this release is already per-ROI z-scored and carries no amplitude to "
                "preserve. Point --scan_dir at an amplitude-bearing release (data/adni/raw)."
            )
        print(f"amplitude check: max/median per-ROI sd on full runs reaches {spread:.3f} "
              f"(a z-scored release gives 1.000)")
        if args.amplitude == "preserve":
            # ONE scalar per scan: removes the arbitrary BOLD gain, leaves every between-ROI
            # amplitude ratio intact -- the whole difference from --amplitude zscore, which
            # divides by 424 different numbers and flattens those ratios to 1. It is NOT
            # upstream's recipe: it also removes between-SUBJECT amplitude, measurable as
            # every scan's median per-ROI sd landing on exactly 1.000.
            keep = [(k[0] / max(k[3], 1e-12), k[1], k[2], k[3], k[4]) for k in keep]
        else:
            print("--amplitude official: mean-centred per ROI, no division of any kind, "
                  "as upstream. Between-subject amplitude is retained.")
            if args.amplitude == "official_offset":
                sys.path.insert(0, str(ROOT / "src"))
                from brainlm_input import fit_voxel_offset
                off = fit_voxel_offset([k[0] for k in keep], source="adni_this_build")
                keep = [(k[0] - off.offset[:, None], k[1], k[2], k[3], k[4]) for k in keep]
                torch.save({"offset": off.offset, "source": off.source},
                           out_dir_for_offset / f"voxel_offset_{args.label_col}.pt")
                print(f"  step 2: subtracted per-ROI median/IQR fitted on this build "
                      f"(mean {off.offset.mean():+.4f}, sd {off.offset.std():.4f}, "
                      f"range [{off.offset.min():+.3f},{off.offset.max():+.3f}])")
        sd_all = torch.cat([k[0].std(1, unbiased=False) for k in keep])
        print(f"amplitude preserved: per-ROI sd over kept scans "
              f"p5 {sd_all.quantile(0.05):.3f} p50 {sd_all.quantile(0.5):.3f} "
              f"p95 {sd_all.quantile(0.95):.3f}  (--amplitude zscore would make all of these 1.000)")

    x = torch.stack([k[0] for k in keep])
    y = torch.tensor([k[1].label for k in keep], dtype=torch.float32)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The tag must name every choice that changes the tensor, or two different datasets
    # collide on one filename and the second build silently overwrites the first.
    tag = (
        f"{scan_dir.name}_{args.label_col}_{args.sessions}"
        + (f"_tr{args.target_tr:g}" if args.target_tr else "")
        + f"_t{args.tlen}"
        + {"preserve": "_amp", "official": "_off", "official_offset": "_offc",
           "zscore": ""}[args.amplitude]
        + (f"_dx{''.join(args.dx_filter)}" if args.dx_filter else "")
        + (f"_{args.protocol}" if args.protocol else "")
        + ("_trhom" if args.exclude_protocol_group else "")
    )
    torch.save(x, out_dir / f"ADNI_x_{tag}.pt")
    torch.save(y, out_dir / f"ADNI_y_{tag}.pt")

    # The manifest carries everything downstream analyses need so they never have to
    # re-derive it: PTID for grouped splits, the acquisition-protocol group for
    # stratified reporting, and the demographic covariates used as a reference
    # baseline (they are NOT model inputs -- see scripts/crossval_adni.py).
    manifest = pd.DataFrame(
        {
            "FILE_NAME": [k[1].FILE_NAME for k in keep],
            "PTID": [k[1].PTID for k in keep],
            "label": [k[1].label for k in keep],
            "n_timepoints_raw": [k[2] for k in keep],
            # Which of the three paths this run took. Load-bearing for the analysis: the
            # path correlates with acquisition protocol, so any effect must be checkable
            # against it rather than silently confounded with it.
            "length_path": [(args.fill if f in {n for n, _, _ in tiled} else
                             "stretch" if f in {n for n, _, _ in stretched} else
                             "pad" if f in {n for n, _, _ in padded} else
                             "resample" if f in {n for n, _, _ in resampled_runs} else
                             "crop") for f in (k[1].FILE_NAME for k in keep)],
            "protocol_group": [protocol_group(k[2]) for k in keep],
            "native_tr": [native_tr(k[2]) for k in keep],
            "target_tr": args.target_tr if args.target_tr else None,
            "window_seconds": args.tlen * (args.target_tr or 0),
            "scan_release": scan_dir.name,
            "amplitude": args.amplitude,
            "gain_divided_out": [k[3] for k in keep],
        }
    )
    for col in ("dx_stage", "batch", "scan_protocol", "COLPROT", "SITE", "AGE", "PTGENDER",
                "PTEDUCAT", "APOE4", "ADNI_MEM", "ADNI_EF"):
        if col in df.columns:
            manifest[col] = [getattr(k[1], col, None) for k in keep]
    if args.label_col in CONTINUOUS:
        # adni_folds.assign_folds stratifies on a binary `label`; a median split of the
        # continuous target keeps the folds balanced on it without changing the target.
        manifest["label_binary_for_folds"] = (
            manifest["label"] > manifest["label"].median()).astype(int)
    manifest.to_csv(out_dir / f"ADNI_manifest_{tag}.csv", index=False)
    print("protocol groups:", manifest.groupby("protocol_group")["label"].agg(["size", "mean"]).to_dict("index"))

    if args.label_col in CONTINUOUS:
        print(f"X={tuple(x.shape)} y={tuple(y.shape)} continuous target "
              f"mean {float(y.mean()):+.3f} sd {float(y.std()):.3f} "
              f"range [{float(y.min()):+.2f},{float(y.max()):+.2f}]")
    elif args.label_col in ORDINAL:
        vals, cnt = np.unique(y.numpy(), return_counts=True)
        print(f"X={tuple(x.shape)} y={tuple(y.shape)} ordinal levels "
              f"{dict(zip(np.round(vals, 3).tolist(), cnt.tolist()))}")
    else:
        pos = int(y.sum())
        print(f"X={tuple(x.shape)} y={tuple(y.shape)} positives={pos} negatives={len(y) - pos}")
    print(f"wrote {out_dir}/ADNI_{{x,y,manifest}}_{tag}.*")


if __name__ == "__main__":
    main()
