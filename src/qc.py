"""Quality-control metrics for parcellated rsfMRI timeseries.

Shared by `scripts/qc_adni.py` (session-level ADNI scans) and `scripts/qc_cohort.py`
(the dual-timepoint ACT / CogTE tensors) so both report the same quantities under the
same definitions.

What the inputs allow. Every tensor in this project is already parcellated to 424 ROIs
and per-ROI z-scored, and no realignment parameters, confound tables or preprocessing
records ship with any of them. Motion is therefore only ever measured by surrogate --
standardised DVARS, the accepted substitute when realignment parameters are unavailable --
and a high value cannot be attributed to head motion rather than physiological or
reconstruction artefact.

Comparability caveat that governs how these numbers may be used: frame-to-frame change
scales with the repetition time, so DVARS, `max_abs` and `lag1_ac` are only comparable
between scans acquired at the same TR. Outlier decisions must be made within an
acquisition group, never against a pooled distribution.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

#: Frames used for connectivity metrics, so FC noise is comparable across scan lengths.
FC_FRAMES = 140

#: Iglewicz-Hoaglin cutoff on a median/MAD scale.
ROBUST_Z = 3.5


def scan_metrics(sig: np.ndarray, fc_frames: int = FC_FRAMES) -> dict:
    """QC metrics for one [R, T] parcellated, per-ROI z-scored timeseries.

    Returns a dict of scalars plus `_edges`, the upper-triangular FC vector, which the
    caller pops to build cohort-level connectome statistics.
    """
    r, t = sig.shape
    finite = np.isfinite(sig)
    out = {
        "n_roi": r,
        "n_time": t,
        "n_nonfinite": int((~finite).sum()),
        "sha1_8": hashlib.sha1(np.ascontiguousarray(sig).tobytes()).hexdigest()[:8],
    }
    clean = np.where(finite, sig, 0.0)

    sd = clean.std(axis=1)
    out["n_constant_roi"] = int((sd < 1e-6).sum())
    out["n_lowvar_roi"] = int((sd < 0.1).sum())

    out["max_abs"] = float(np.abs(clean).max())
    out["frac_gt3"] = float((np.abs(clean) > 3).mean())
    out["frac_gt5"] = float((np.abs(clean) > 5).mean())

    # Standardised DVARS: RMS across ROIs of the frame-to-frame difference.
    d = np.diff(clean, axis=1)
    dvars = np.sqrt((d ** 2).mean(axis=0))
    out["dvars_mean"] = float(dvars.mean())
    out["dvars_p95"] = float(np.percentile(dvars, 95))
    out["dvars_max"] = float(dvars.max())
    # Spikiness relative to the scan's own median, so this one is TR-independent.
    med = float(np.median(dvars))
    out["frac_spiky_frames"] = float((dvars > 1.5 * med).mean()) if med > 0 else 1.0

    gs = clean.mean(axis=0)
    out["gs_std"] = float(gs.std())
    out["gs_max_abs"] = float(np.abs(gs).max())

    a, b = clean[:, :-1], clean[:, 1:]
    denom = a.std(axis=1) * b.std(axis=1)
    ac = np.where(denom > 0, (a * b).mean(axis=1) / np.maximum(denom, 1e-12), 0.0)
    out["lag1_ac"] = float(ac.mean())

    crop = clean[:, :fc_frames]
    fc = np.nan_to_num(np.corrcoef(crop), nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices(r, k=1)
    edges = fc[iu]
    out["fc_mean"] = float(edges.mean())
    out["fc_mean_abs"] = float(np.abs(edges).mean())
    out["frac_fc_gt09"] = float((np.abs(edges) > 0.9).mean())
    out["_edges"] = edges.astype(np.float32)
    return out


def robust_z(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - med) / mad


def leave_one_out_typicality(edges: np.ndarray) -> np.ndarray:
    """Correlation of each row's FC vector with the mean of all the others.

    Leave-one-out so a single gross outlier cannot inflate its own score by dominating
    the reference mean.
    """
    n = len(edges)
    total = edges.sum(axis=0)
    typ = np.empty(n)
    for i in range(n):
        loo = (total - edges[i]) / max(n - 1, 1)
        a, b = edges[i] - edges[i].mean(), loo - loo.mean()
        typ[i] = float((a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return typ


#: Metrics whose extremes are judged within an acquisition group.
FLAG_METRICS = ("dvars_mean", "dvars_max", "frac_spiky_frames", "max_abs",
                "gs_std", "frac_fc_gt09")


def add_flags(qc: pd.DataFrame, group_col: str, robust_z_cut: float = ROBUST_Z) -> pd.DataFrame:
    """Add robust-z columns, `n_outlier_metrics` and `qc_flag`, grouped by `group_col`."""
    qc = qc.copy()
    for col in FLAG_METRICS:
        qc[f"rz_{col}"] = qc.groupby(group_col)[col].transform(robust_z)
    qc["rz_fc_typicality"] = qc.groupby(group_col)["fc_typicality"].transform(robust_z)
    qc["n_outlier_metrics"] = sum(
        (qc[f"rz_{c}"].abs() > robust_z_cut).astype(int) for c in FLAG_METRICS
    )
    qc["qc_flag"] = (
        (qc["n_nonfinite"] > 0)
        | (qc["n_constant_roi"] > 0)
        | qc["sha1_8"].duplicated(keep=False)
        | (qc["rz_fc_typicality"] < -robust_z_cut)
        | (qc["n_outlier_metrics"] >= 2)
    )
    return qc


def flag_rule_text(robust_z_cut: float = ROBUST_Z) -> str:
    return (
        "non-finite values OR a constant ROI OR duplicate scan content OR an atypical "
        f"connectome (robust z < -{robust_z_cut}) OR >=2 metrics with |robust z| > {robust_z_cut}, "
        "all judged within acquisition group"
    )
