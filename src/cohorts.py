"""Load an RCT cohort: imaging joined to its outcome table by subject ID.

Before the 2026-08 data refresh, `bold[i]` and `labels.csv` row `i` were assumed to be
the same subject. That assumption is now false in both directions -- the refresh added 5
ACT subjects that the outcome table does not cover and dropped 2 CogTE subjects that it
does -- so the join happens on the subject ID and both kinds of mismatch are reported
rather than silently absorbed.

Row order follows `labels.csv`, restricted to subjects that have imaging. Keeping the
table's order means the shipped `fold_id_*` columns keep pointing at the same subjects
they were stratified for.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import valid_lengths
from paths import COHORT
from temporal import describe as describe_tr


@dataclass
class Cohort:
    """Imaging + outcomes for one cohort, aligned row-for-row."""

    name: str
    subjects: np.ndarray                  # [N] subject IDs, in row order
    bold: torch.Tensor                    # [N, 424, T, 2], NaN-padded as stored
    valid: torch.Tensor                   # [N, 2] usable frames per scan
    native_tr: float | None
    bold_path: Path
    table: pd.DataFrame | None = None     # N rows, same order as `subjects`
    unlabelled: list[str] = field(default_factory=list)   # imaged, not in the table
    unimaged: list[str] = field(default_factory=list)     # in the table, not imaged

    def __len__(self) -> int:
        return len(self.subjects)

    def labels(self, label: str) -> np.ndarray:
        """The binary outcome for `label` ('mem' / 'ef') as float, one per row.

        SIGN: the CSV columns are named `MEM_decline_bi` / `EF_decline_bi`, but **1 means
        the subject IMPROVED**, not declined. Verified against the underlying change
        scores in both labelled cohorts:

            ACT    MEM_decline_bi=1 -> mean delay_recall_ch  +6.64   (=0 -> -9.98)
            ACT    EF_decline_bi =1 -> mean exe_composite_ch +0.357  (=0 -> -0.280)
            CogTE  MEM_decline_bi=1 -> mean delay_recall_ch  +7.91   (=0 -> -10.11)
            CogTE  EF_decline_bi =1 -> mean exe_composite_ch +0.236  (=0 -> -0.250)

        So the positive class is cognitive improvement over the trial, which is the
        quantity of interest. AUC is NOT invariant to this: reading the column name
        literally reports every result as 1-AUC of what it actually measures.
        `check_label_sign` asserts it at load time so a data refresh cannot flip it
        silently.
        """
        col = COHORT[self.name]["label_col_in_csv"][label]
        return self._column(col, allow_missing=True).astype(float)

    def change_scores(self, label: str) -> np.ndarray:
        """The CONTINUOUS change score underlying `labels(label)`, one per row.

        Positive = improvement, same orientation as `labels`. Binarising throws away the
        magnitude, which matters at n=95/76; this exposes the regression target.
        """
        col = COHORT[self.name]["change_col_in_csv"][label]
        return self._column(col, allow_missing=True).astype(float)

    def baseline_levels(self, label: str) -> np.ndarray:
        """The BASELINE (pre-intervention) level of `label`'s outcome, one per row.

        A level and its change score are not equally measurable. The change is a difference
        of two noisy scores, so it carries roughly twice the measurement-error variance of
        either one, and this project measured the consequence directly: the pooled
        displacement test needs n ~ 1086 to detect the effect it observed, against 274
        available. The level is the same construct without that penalty, so it is the
        reference that says whether a representation carries ANY cognition-relevant signal
        in these cohorts before change is asked of it.
        """
        col = COHORT[self.name]["level_col_in_csv"][label]
        return self._column(col, allow_missing=True).astype(float)

    def followup_levels(self, label: str) -> np.ndarray:
        """The FOLLOW-UP (post-intervention) level of `label`'s outcome, one per row.

        The T2 counterpart of `baseline_levels`. Pairing T1 features against T1 cognition
        with T2 features against T2 cognition measures the same cross-sectional question
        twice in the same subjects, which is the internal replication the cross-sectional
        arm of this study was missing: a level correlation that appears at one session and
        not the other is a session effect, not a representation reading cognition.

        It is NOT a prediction of the future. T2 features and T2 cognition are concurrent,
        exactly as T1 features and T1 cognition are.
        """
        col = COHORT[self.name]["level_t2_col_in_csv"][label]
        return self._column(col, allow_missing=True).astype(float)

    def check_label_sign(self, label: str) -> float:
        """Assert that label 1 really is the improving group. Returns the mean difference.

        Raises if the orientation ever flips -- e.g. after a data refresh that renames or
        recodes the outcome columns.
        """
        m = self.labelled_mask(label)
        y = self.labels(label)[m]
        ch = self.change_scores(label)[m]
        d = float(np.nanmean(ch[y == 1]) - np.nanmean(ch[y == 0]))
        if not d > 0:
            raise ValueError(
                f"{self.name}/{label}: label 1 has mean change {d:+.3f} relative to label 0, "
                "so the positive class is NOT the improving group any more. Every AUC "
                "computed from this column would be reported with the wrong sign."
            )
        return d

    def folds(self, fold_col: str) -> np.ndarray:
        """Fold ids. May contain NaN where a row is not scorable -- see labelled_mask."""
        return self._column(fold_col, allow_missing=True)

    def labelled_mask(self, label: str) -> np.ndarray:
        """Boolean mask of rows this cohort can actually be scored on for `label`.

        ACT and CogTE are complete, so this is all-True there. BREATHE is not: its two
        outcomes come from different source tables with different follow-up coverage
        (MEM 105/107 imaged subjects, EF 96/107). Scoring the missing rows as if they were
        zeros or dropping them silently would both be wrong, so callers subset on this.
        """
        meta = COHORT[self.name]
        cols = [meta["label_col_in_csv"].get(label)]
        if label in meta.get("change_col_in_csv", {}):
            cols.append(meta["change_col_in_csv"][label])
        # A row with no fold assignment cannot enter cross-validation either.
        if label in meta.get("fold_col_for_label", {}):
            cols.append(meta["fold_col_for_label"][label])
        ok = np.ones(len(self), dtype=bool)
        for c in cols:
            if c and self.table is not None and c in self.table.columns:
                ok &= self.table[c].notna().values
        return ok & self.analysis_mask()

    def analysis_mask(self) -> np.ndarray:
        """Cohort-level exclusions only, independent of any outcome's completeness.

        BREATHE uses this to drop the 26 non-randomised healthy controls folded into its
        arm 1 (see the `exclude_where` note in paths.COHORT). Arm contrasts want exactly
        this and NOT `labelled_mask`: an arm is known for every randomised participant, so
        requiring a cognitive outcome as well would discard usable subjects and make the
        randomisation analysis depend on outcome availability.
        """
        meta = COHORT[self.name]
        ok = np.ones(len(self), dtype=bool)
        ex = meta.get("exclude_where")
        if ex is not None:
            col, bad = ex
            if self.table is None or col not in self.table.columns:
                raise KeyError(
                    f"cohort {self.name} declares exclude_where={ex!r} but {col!r} is not "
                    f"in {meta['labels_csv'].name}. Rebuild the label table -- silently "
                    f"skipping the exclusion would put excluded subjects back in.")
            ok &= (self.table[col].values != bad)
        return ok

    def arm_contrast(self, name: str) -> np.ndarray:
        """Binary randomised-arm label for contrast `name`: 1.0, 0.0, or NaN if excluded.

        These are the only NOISELESS outcomes in the trials. A cognitive change score is
        attenuated by its own test-retest unreliability (the observed correlation scales as
        sqrt(reliability)), whereas arm assignment is known exactly -- so an arm contrast
        buys power that no change to the encoder can.

        The contrast must be scored on the T2 - T1 CHANGE. Arm is randomised, so baseline
        imaging cannot predict it by construction; that makes the baseline version a free
        negative control which MUST come out null. If it does not, arm is confounded with
        site, scanner drift or a labelling error.
        """
        meta = COHORT[self.name]
        spec = meta.get("arm_contrasts", {}).get(name)
        if spec is None:
            avail = sorted(meta.get("arm_contrasts", {}))
            raise KeyError(
                f"cohort {self.name} has no arm contrast {name!r} (available: {avail}). "
                f"For BREATHE the arm codes are known but their meaning is not recorded in "
                f"any source table, so no contrast is defined rather than guessed.")
        arm = self._column(meta["arm_col"], allow_missing=True)
        y = np.full(len(self), np.nan)
        for value, codes in spec.items():
            y[np.isin(arm, codes)] = float(value)
        return y

    def _column(self, col: str, allow_missing: bool = False) -> np.ndarray:
        if self.table is None:
            raise ValueError(f"cohort {self.name} has no outcome table")
        if col not in self.table.columns:
            raise KeyError(f"{col!r} not in {COHORT[self.name]['labels_csv']}")
        v = self.table[col].values
        if pd.isna(v).any() and not allow_missing:
            raise ValueError(
                f"{col!r} has {int(pd.isna(v).sum())} missing values. If that is expected "
                f"for this cohort, subset on Cohort.labelled_mask({col!r}) first.")
        return v

    def report(self) -> str:
        lines = [
            f"cohort {self.name}: n={len(self)} scans=2 (T1,T2) from {self.bold_path.name}",
            f"  usable frames: T1 {int(self.valid[:, 0].min())}-{int(self.valid[:, 0].max())}, "
            f"T2 {int(self.valid[:, 1].min())}-{int(self.valid[:, 1].max())} "
            f"of {self.bold.shape[2]} stored",
        ]
        if self.native_tr:
            lines.append("  " + describe_tr(self.native_tr, int(self.valid.min())))
        else:
            lines.append("  native TR unknown for this cohort -- pass --native_tr")
        if self.unlabelled:
            lines.append(f"  {len(self.unlabelled)} imaged subjects have no outcome row, "
                         f"excluded: {list(map(str, self.unlabelled))}")
        if self.unimaged:
            lines.append(f"  {len(self.unimaged)} subjects in the outcome table have no "
                         f"imaging in this release, excluded: {list(map(str, self.unimaged))}")
        return "\n".join(lines)


def center_rois(co: "Cohort") -> "Cohort":
    """Subtract each ROI's own temporal mean, per scan, over that scan's VALID frames.

    Why this is not optional. The ADNI build the transfer axis is fitted on is
    `--amplitude official`, which is BrainLM's upstream recipe: per-ROI mean-centring and
    no division of any kind (`prepare_adni.py`, `sig = sig - sig.mean(1, keepdim=True)`).
    `load_cohort` deliberately applies no normalisation, and `rct_probe.features()` crops
    without centring, so before this function existed the RCT side reached the encoders in
    six different states -- and only three of the six matched ADNI, by accident:

        A424 `amp`   ACT un-centred, BREATHE un-centred, CogTE centred via upsample_short
        Schaefer     ACT already centred upstream, BREATHE un-centred, CogTE via upsample_short

    ACT's per-ROI offset in the A424 release has median |mean| 0.956 against a per-ROI sd of
    0.736 -- larger than the signal. Measured consequence on the decodability control
    (BrainLM 13M, roi_norm L4, tlen 200), un-centred -> centred:

        ACT    sex 0.5900 -> 0.6899   age 0.1800 -> 0.3885   site 0.4037 -> 0.6486
        CogTE  sex 0.8022 -> 0.8022   age 0.4819 -> 0.4819       (bit-identical: already centred)

    CogTE being unchanged to four decimals is the internal control that identifies the
    offset as the cause rather than something else about centring.

    The per-ROI temporal mean of a BOLD parcel is acquisition baseline -- tissue
    properties, coil sensitivity, scaling convention -- not neural signal, and it differs
    between sessions of the same subject, so keeping it lands the nuisance directly in the
    T2 - T1 difference that every longitudinal analysis rests on.

    Centring is NOT z-scoring: each ROI keeps its own variance, so between-ROI amplitude --
    which out-predicts the encoders on ADNI and is load-bearing in this project -- survives
    intact. No division is applied here, matching `--amplitude official` rather than
    `--amplitude preserve`.

    Baselines are unaffected by construction: FC is a correlation and the amplitude profile
    is a ratio of standard deviations, both invariant to a per-ROI offset. Only the encoder
    rows move.
    """
    b = co.bold.clone().float()
    v = co.valid.numpy()
    for tp in range(b.shape[3]):
        for i in range(b.shape[0]):
            n = int(v[i, tp])
            if n < 1:
                continue
            blk = b[i, :, :n, tp]
            b[i, :, :n, tp] = blk - blk.mean(dim=1, keepdim=True)
    return replace(co, bold=b)


def cohort_tr(name: str) -> float | None:
    """The cohort's TR: `meta.json` in the cohort directory wins over temporal.COHORT_TR.

    A cohort whose TR is not in the code table (BREATHE) becomes self-describing by
    dropping `{"native_tr": <seconds>}` next to its tensor, with no code change and no
    per-run --native_tr to remember.
    """
    meta_file = Path(COHORT[name]["dir"]) / "meta.json"
    if meta_file.exists():
        tr = json.loads(meta_file.read_text()).get("native_tr")
        if tr is not None:
            return float(tr)
    return COHORT[name]["native_tr"]


def load_cohort(name: str, require_labels: bool = True,
                release: str = "amp") -> Cohort:
    """Read `data/rct/<name>/` and align imaging to outcomes by subject ID.

    `release` picks WHICH FILE to read, and nothing else -- no normalisation is applied
    here:

        "amp"    <cohort>_A424_T1T2_raw_combined.pt   amplitude intact (2026-08-24)
        "zscore" bold_T1T2.pt                         per-ROI z-scored (2026-08-19)

    psopt additionally ships the same recordings under two DENOISINGS, delivered
    2026-09-17 as `pSOPT_visits12_GSR/` and `pSOPT_visits12_noGSR/`, selectable as
    `amp` / `amp_gsr` (and the zscore / schaefer counterparts). `amp` is the
    noGSR one, because act/cogte/breathe are noGSR (user, 2026-09-17). MEASURED, and it
    does not follow from the label: the delivered regression is PARTIAL. global-signal/per-ROI-sd is
    0.720 in the noGSR build and only 0.527 in the GSR one, against 0.152 (act) / 0.248
    (cogte) / 0.152 (breathe) and 0.351 for ADNI's own build; applying a full regression
    on top of the delivered GSR build still drives it to 4.5e-08, so a large removable
    component remains. Neither psopt build is on the other cohorts' SCALE either -- both
    are raw parcel intensity, per-ROI sd 5.25 (GSR) / 6.23 (noGSR) against 0.68-0.73.

    They are the same recordings; z-scoring the first reproduces the second to 1.3e-04
    (ACT), 1.3e-05 (CogTE), 4.4e-05 (BREATHE). The pair is the control in which only
    amplitude differs.

    The argument used to be called `amplitude`, which collided with
    `prepare_adni.py --amplitude`. That flag names a TRANSFORM applied to ADNI scans
    (zscore / preserve / official / official_offset); this one only names a file. Same word,
    unrelated meanings -- hence the rename.
    """
    RELEASES = ("amp", "zscore", "schaefer",
                "amp_gsr", "zscore_gsr", "schaefer_gsr")
    if release not in RELEASES:
        raise ValueError(f"release must be one of {RELEASES}, got {release!r}")
    meta = COHORT[name]
    key = {"amp": "bold_amp", "zscore": "bold_zscore",
           "schaefer": "bold_schaefer",
           "amp_gsr": "bold_amp_gsr", "zscore_gsr": "bold_zscore_gsr",
           "schaefer_gsr": "bold_schaefer_gsr"}[release]
    if key not in meta:
        raise ValueError(
            f"cohort {name!r} has no {release!r} release. The *_gsr releases exist only "
            "for psopt, which ships two denoisings of the same recordings "
            "(pSOPT_visits12_GSR and pSOPT_visits12_noGSR).")
    bold_path = Path(meta[key])
    if not bold_path.exists():
        raise FileNotFoundError(
            f"cohort {name}: release={release!r} needs {bold_path}, which is absent")
    # The Schaefer release does not always cover every subject the A424 release does
    # (CogTE ships 66 of 74), so it carries its own subject list when one exists.
    subj_path = meta["subjects"]
    if release == "schaefer" and meta.get("subjects_schaefer") is not None:
        subj_path = meta["subjects_schaefer"]
    for p in (bold_path, subj_path):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    bold = torch.load(bold_path, map_location="cpu", weights_only=False)
    subjects = np.array([s for s in Path(subj_path).read_text().split() if s])
    if len(subjects) != bold.shape[0]:
        raise ValueError(
            f"{subj_path} lists {len(subjects)} subjects but {bold_path.name} has "
            f"{bold.shape[0]} rows"
        )
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"{subj_path} contains duplicate subject IDs")

    csv_path = Path(meta["labels_csv"])
    if not csv_path.exists():
        if require_labels:
            raise FileNotFoundError(
                f"{csv_path} -- cohort {name} has imaging but no outcome table, so it "
                "cannot be used for supervised training or scoring"
            )
        return Cohort(name, subjects, bold, valid_lengths(bold), cohort_tr(name), bold_path)

    table = pd.read_csv(csv_path)
    id_col = meta["id_col"]
    if id_col not in table.columns:
        raise KeyError(f"{csv_path} has no {id_col!r} column")
    if table[id_col].duplicated().any():
        raise ValueError(f"{csv_path} has duplicate {id_col} values")

    imaged = {s: i for i, s in enumerate(subjects)}
    in_both = table[id_col].isin(imaged).values
    table = table[in_both].reset_index(drop=True)
    rows = torch.as_tensor([imaged[s] for s in table[id_col]], dtype=torch.long)
    bold = bold[rows]
    kept = set(table[id_col])

    return Cohort(
        name=name,
        subjects=table[id_col].astype(str).values,
        bold=bold,
        valid=valid_lengths(bold),
        native_tr=cohort_tr(name),
        bold_path=bold_path,
        table=table,
        unlabelled=sorted(map(str, set(subjects) - kept)),
        unimaged=sorted(map(str, set(pd.read_csv(csv_path)[id_col]) - kept)),
    )
