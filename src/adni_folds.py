"""Covariate-stratified, subject-grouped fold assignment for ADNI.

`StratifiedGroupKFold` balances one label. Diagnosis is not the only thing that needs
balancing here: age, sex and education each predict AD in this cohort on their own
(demographics alone reach AUC 0.7498, the original cohort audit 4b-2), so a fold that happens
to be older or less educated will look easier or harder for reasons that have nothing to
do with the scans.

Approach: build a per-subject stratum from diagnosis x sex x age bin x education bin,
then deal the subjects of each stratum round-robin into folds. Dealing rather than random
assignment keeps small strata spread evenly instead of clumping, and the whole assignment
is done on subjects, so no participant can span a fold boundary regardless of how many
sessions they contribute.

Two details make the full crossing usable at n=570 with 113 positives:

  * A cell smaller than `n_splits` is fine. Dealing it round-robin just leaves some folds
    without a member of that cell. Coarsening is triggered only when a large share of
    subjects sit in cells of size one, where dealing has no freedom left. (Requiring every
    cell to hold `n_splits` members discarded education over a single 4-subject cell.)
  * Inside a cell, subjects are dealt in education-then-age order rather than at random,
    so the continuous covariates spread evenly across folds. Measured on the 570-subject
    set, this cuts the across-fold spread in mean education from 0.57 to 0.07-0.16 years
    while keeping the label rate within 0.014-0.025.

`balance_report` prints the achieved per-fold balance so it can be checked rather than
assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Covariates dropped in this order when strata are too thin to deal across folds.
COARSEN_ORDER = ("edu_bin", "age_bin", "sex")


def _bin_covariates(subjects: pd.DataFrame, age_bins: int, edu_bins: int) -> pd.DataFrame:
    out = subjects.copy()
    out["sex"] = (out["PTGENDER"].astype(str).str.lower() == "male").astype(int)
    for col, src, n in (("age_bin", "AGE", age_bins), ("edu_bin", "PTEDUCAT", edu_bins)):
        v = pd.to_numeric(out[src], errors="coerce")
        # Median-impute the rare missing value so it does not become its own stratum.
        v = v.fillna(v.median())
        # Quantile bins, deduplicated in case the distribution is too discrete to split.
        out[col] = pd.qcut(v, n, labels=False, duplicates="drop").astype(int)
    return out


def assign_folds(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    age_bins: int = 3,
    edu_bins: int = 2,
    max_singleton_frac: float = 0.10,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return (fold_id per row of `manifest`, info dict).

    `manifest` needs PTID, label, AGE, PTGENDER, PTEDUCAT. Rows are sessions; folds are
    assigned to subjects and broadcast back to their sessions.
    """
    needed = {"PTID", "label", "AGE", "PTGENDER", "PTEDUCAT"}
    missing = needed - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing {sorted(missing)}")

    # One row per subject; label and covariates are subject-level constants here, so the
    # first session is representative.
    subjects = manifest.drop_duplicates("PTID").set_index("PTID")
    subjects = _bin_covariates(subjects, age_bins, edu_bins)

    keys = ["label", "sex", "age_bin", "edu_bin"]
    dropped = []
    # A stratum smaller than n_splits is not a problem -- dealing it round-robin simply
    # leaves some folds without a member of that cell. What does break balance is a large
    # share of subjects sitting in cells of size one, where dealing has no freedom left.
    # So coarsen on that condition, not on min-cell-size. (The earlier all-or-nothing
    # rule discarded education over a single 4-subject cell.)
    while True:
        strata = subjects[keys].astype(str).agg("|".join, axis=1)
        counts = strata.value_counts()
        singleton_frac = counts[counts == 1].sum() / len(subjects)
        if singleton_frac <= max_singleton_frac or len(keys) == 1:
            break
        drop = next((c for c in COARSEN_ORDER if c in keys), None)
        if drop is None:
            break
        keys = [k for k in keys if k != drop]
        dropped.append(drop)

    rng = np.random.default_rng(seed)
    fold_of_subject = {}
    # Within a stratum, deal in order of the continuous covariates rather than at random.
    # Round-robin over an education-then-age-sorted list spreads those variables evenly
    # across folds even when they are not part of the crossed key, which is what keeps
    # the achieved balance tight without exploding the number of cells.
    order_cols = [c for c in ("PTEDUCAT", "AGE") if c in subjects.columns]
    for _, members in sorted(strata.groupby(strata), key=lambda kv: kv[0]):
        block = subjects.loc[members.index, order_cols].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        )
        # Random tie-break so repeated seeds give different, still-balanced assignments.
        block = block.assign(_jitter=rng.random(len(block)))
        ids = block.sort_values([*order_cols, "_jitter"]).index.to_numpy()
        offset = rng.integers(0, n_splits)
        for i, sid in enumerate(ids):
            fold_of_subject[sid] = int((i + offset) % n_splits) + 1

    folds = manifest["PTID"].map(fold_of_subject).to_numpy()
    if np.isnan(folds.astype(float)).any():
        raise RuntimeError("some sessions did not receive a fold")

    info = {
        "stratify_on": keys,
        "coarsened_away": dropped,
        "n_strata": int(strata.nunique()),
        "min_stratum": int(strata.value_counts().min()),
        "n_subjects": int(len(subjects)),
        "n_sessions": int(len(manifest)),
    }
    if verbose:
        print(
            f"folds: stratified on {'+'.join(keys)}"
            + (f" (coarsened away {'+'.join(dropped)})" if dropped else "")
            + f" | {info['n_strata']} strata, smallest {info['min_stratum']}"
            + f" | {info['n_subjects']} subjects -> {info['n_sessions']} sessions",
            flush=True,
        )
        print(balance_report(manifest, folds).to_string(), flush=True)
    return folds.astype(int), info


def balance_report(manifest: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Per-fold covariate balance, so the stratification can be checked rather than assumed."""
    df = manifest.copy()
    df["fold"] = folds
    df["male"] = (df["PTGENDER"].astype(str).str.lower() == "male").astype(float)
    rows = []
    for fold, sub in df.groupby("fold"):
        subj = sub.drop_duplicates("PTID")
        rows.append(
            {
                "fold": int(fold),
                "subjects": len(subj),
                "sessions": len(sub),
                "AD_rate": round(float(subj["label"].mean()), 4),
                "age_mean": round(float(pd.to_numeric(subj["AGE"], errors="coerce").mean()), 2),
                "male_frac": round(float(subj["male"].mean()), 4),
                "edu_mean": round(float(pd.to_numeric(subj["PTEDUCAT"], errors="coerce").mean()), 2),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def aggregate_by_subject(manifest: pd.DataFrame, scores: np.ndarray, labels: np.ndarray):
    """Mean prediction per subject, so participants with more sessions are not weighted more.

    Returns (subject_ids, subject_labels, subject_scores).
    """
    df = pd.DataFrame({"PTID": manifest["PTID"].to_numpy(), "score": scores, "label": labels})
    g = df.groupby("PTID", sort=True).agg(score=("score", "mean"), label=("label", "first"))
    return g.index.to_numpy(), g["label"].to_numpy(), g["score"].to_numpy()
