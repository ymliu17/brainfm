#!/usr/bin/env python3
"""Build `data/rct/breathe/labels.csv` in the ACT / CogTE schema.

BREATHE shipped imaging and a cognition table but no outcome table, so it could be encoded
and QC'd but never trained or scored. This assembles the outcomes.

OUTCOME DEFINITIONS
-------------------
EF  comes straight from `breathe_ef_responder_4task.csv`: `ef_change` (continuous) and
    `ef_responder` (binary). Verified against that file: `ef_responder == (ef_change > 0)`
    for 107/107 rows and `ef_change == ef_post - ef_pre` exactly, so the orientation
    matches ACT / CogTE -- **1 means the subject improved.**

EM  uses `bvmtr_delayrecall_tscore`, chosen to match what ACT and CogTE actually did
    rather than by convention. Verified numerically against the shipped label files:

        ACT    delay_recall_ch == calc_delay_recall_tscore T2-T1   95/95 exact
        CogTE  delay_recall_ch == DelayRecall_tscore      2.0-1.0  76/76 exact

    Both are single delayed-recall t-scores (baseline mean 42.2 / 38.9, sd 11.9 / 13.2), so
    a single BVMT-R delayed-recall t-score is the item-for-item match: same construct, same
    scale, same rule. Two alternatives were measured and rejected:

      - `bvmtr_learn_tscore`, alone or in a composite. Its session-1 to session-2
        correlation is 0.217 (vs 0.731 for delayed recall, 0.724 for total recall) and its
        change-score sd EXCEEDS its baseline sd (ratio 1.251 vs ~0.75), the signature of a
        measure dominated by noise -- it is a slope, so differencing it across sessions
        differences twice. Including it flips 31% of the binary labels relative to the
        alternatives. Decisive point: CogTE's table carries `Learning_tscore` at all four
        timepoints, so learning WAS available there and the shipped labels did not use it.
      - averaging in `bvmtr_totrecall_tscore`. Comparably reliable (0.724) but it
        correlates 0.875 with delayed recall, so it adds little, and it moves BREATHE off
        the single-delayed-recall definition the other two cohorts use.

THE TIME WINDOW
---------------
The cognition table has four sessions; the EF file only says "pre" and "post". The window
was recovered rather than assumed, by rebuilding the 4-task EF composite at each session
and correlating it with `ef_pre` / `ef_post`:

                  sess 1      sess 2      sess 3      sess 4
    ef_pre       +0.9248     +0.5541     +0.6326     +0.4795
    ef_post      +0.7678     +0.9417     +0.7819     +0.5514

The diagonal dominates by a wide margin, so **pre = session 1, post = session 2**. EM uses
the same window, which it must: measuring the two constructs over different intervals would
make them incomparable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paths import COHORT  # noqa: E402

BREATHE = COHORT["breathe"]["dir"]
COG_CSV = BREATHE / "breathe_cognition0323.csv"
EF_CSV = BREATHE / "breathe_ef_responder_4task.csv"

#: The EM measure. A list so a composite can be substituted, but the default is the single
#: delayed-recall t-score that matches ACT / CogTE item-for-item. Components are
#: standardised on the session-1 distribution before averaging; with one component that is
#: a monotone rescaling and changes no label.
EM_COMPONENTS = ["bvmtr_delayrecall_tscore"]
PRE_SESSION, POST_SESSION = 1, 2


def imaging_id(subject_id: str) -> str:
    """`s101` in the cognition tables -> `sub-101` in subjects.txt."""
    return "sub-" + str(subject_id).lstrip("s")


def assign_folds(df: pd.DataFrame, label_col: str, n_folds: int, seed: int,
                 strat_extra: str | None) -> np.ndarray:
    """Fold ids stratified on the outcome (and the trial arm, when present).

    Stratifying on the arm matters in a 3-arm trial with n<100: an unstratified split can
    put most of one arm in a single fold, and arm is associated with the outcome by design.
    """
    from sklearn.model_selection import StratifiedKFold

    key = df[label_col].astype(str)
    if strat_extra and strat_extra in df.columns:
        key = key + "_" + df[strat_extra].astype(str)
    counts = key.value_counts()
    # A stratum smaller than n_folds cannot be spread across folds; merge those.
    key = key.where(key.map(counts) >= n_folds, other="_rare")
    folds = np.zeros(len(df), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(skf.split(np.zeros(len(df)), key), start=1):
        folds[te] = i
    return folds


def main():
    ap = argparse.ArgumentParser(description="Build BREATHE labels.csv")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(BREATHE / "labels.csv"))
    ap.add_argument("--em_components", nargs="+", default=None,
                    help="override the EM components (default bvmtr_delayrecall_tscore). "
                         "For sensitivity analysis only -- note that ravlt_ltpr is measured "
                         "at session 1 ONLY (134/0/0/0), so no verbal-memory change score "
                         "exists in this cohort and the EM construct is necessarily "
                         "visuospatial.")
    ap.add_argument("--dry_run", action="store_true", help="report and write nothing")
    args = ap.parse_args()

    components = args.em_components or EM_COMPONENTS
    cog = pd.read_csv(COG_CSV, low_memory=False)
    ef = pd.read_csv(EF_CSV)
    imaged = [ln.strip() for ln in (BREATHE / "subjects.txt").read_text().splitlines() if ln.strip()]

    missing = [c for c in components if c not in cog.columns]
    if missing:
        raise SystemExit(f"{COG_CSV.name} is missing {missing}")

    pre = cog[cog["session_num"] == PRE_SESSION].set_index("subject_id")
    post = cog[cog["session_num"] == POST_SESSION].set_index("subject_id")
    if pre.index.duplicated().any() or post.index.duplicated().any():
        raise SystemExit("duplicate subject rows within a session")

    # Standardise on session 1 so both subscales weigh equally and session 2 lands on the
    # session-1 scale. Using each session's own mean/sd would destroy the change signal.
    em = {}
    for tag, tbl in (("pre", pre), ("post", post)):
        z = [(tbl[c] - pre[c].mean()) / pre[c].std() for c in components]
        em[tag] = pd.concat(z, axis=1).mean(axis=1, skipna=False)

    df = pd.DataFrame({"subject_id": sorted(set(pre.index) | set(post.index))})
    df["ID"] = df["subject_id"].map(imaging_id)
    df["em_pre"] = df["subject_id"].map(em["pre"])
    df["em_post"] = df["subject_id"].map(em["post"])
    df["em_change"] = df["em_post"] - df["em_pre"]

    efi = ef.set_index("subject_id")
    for c in ("ef_pre", "ef_post", "ef_change", "ef_responder"):
        df[c] = df["subject_id"].map(efi[c]) if c in efi.columns else np.nan

    # Verify the EF orientation we are inheriting rather than trusting the column name.
    k = df.dropna(subset=["ef_change", "ef_responder"])
    agree = int((k["ef_responder"] == (k["ef_change"] > 0).astype(float)).sum())
    if agree != len(k):
        raise SystemExit(f"ef_responder != (ef_change > 0) for {len(k) - agree} of {len(k)} "
                         "rows; the binary EF outcome's orientation is not what it appears")
    print(f"EF orientation check: ef_responder == (ef_change > 0) for {agree}/{len(k)} rows")

    # Schema names matched to ACT / CogTE. The historical columns are called *_decline_bi
    # but encode IMPROVEMENT there too (see cohorts.Cohort.labels), so the same names carry
    # the same meaning here: 1 = improved.
    #
    # THRESHOLD: `>= 0`, not `> 0`. Recovered from the shipped files rather than assumed --
    # `MEM_decline_bi == (change >= 0)` holds 95/95 in ACT and 76/76 in CogTE, whereas
    # `> 0` holds only 84/95 and 60/76. The gap is entirely subjects whose change is
    # exactly 0 (11 in ACT, 16 in CogTE), all labelled 1, i.e. "did not decline". Ties
    # occur because these are integer t-scores; the EF composites are continuous and have
    # no ties, which is why `>` and `>=` agree there. BREATHE's delayed recall is an
    # integer t-score, so this distinction bites here too.
    df["MEM_decline_bi"] = (df["em_change"] >= 0).astype("Int64")
    df.loc[df["em_change"].isna(), "MEM_decline_bi"] = pd.NA
    df["EF_decline_bi"] = df["ef_responder"].astype("Int64")
    df["delay_recall_ch"] = df["em_change"]
    df["exe_composite_ch"] = df["ef_change"]

    pre = cog[cog["session_num"] == PRE_SESSION].drop_duplicates("subject_id") \
             .set_index("subject_id")
    df["group_assignment_T1"] = df["subject_id"].map(pre["randomization"])
    # BREATHE's arm 1 is NOT one intervention: it mixes 26 randomised participants with 26
    # non-randomised healthy controls, and `cognitive_status` is the only column that
    # separates them (`mci_diagnosis` does not -- arm 1 reads 42/10 on it against 14/13 and
    # 15/13 for arms 2 and 3, and becomes a consistent 16/10 only after the HC are removed).
    # The HC are 5.7 years younger (68.7 vs 74.4), more often female (0.77 vs 0.50) and
    # nearly an SD higher on baseline EF (0.61 vs 0.08), so leaving them in inflates
    # between-subject variance on exactly the covariates the positive control reads.
    # Dropping cognitive_status == 0 leaves the balanced 26 / 27 / 28 a three-arm
    # randomisation should have.
    df["cognitive_status"] = df["subject_id"].map(pre["cognitive_status"])

    df["imaged"] = df["ID"].isin(imaged)
    print(f"\ncognition subjects {len(df)}   imaged {df['imaged'].sum()}/{len(imaged)}")
    for lab, ch in (("MEM_decline_bi", "delay_recall_ch"), ("EF_decline_bi", "exe_composite_ch")):
        u = df[df["imaged"] & df[lab].notna()]
        d = float(u.loc[u[lab] == 1, ch].mean() - u.loc[u[lab] == 0, ch].mean())
        if not d > 0:
            raise SystemExit(f"{lab}: label 1 has mean change {d:+.3f} vs label 0, so the "
                             "positive class is not the improving group")
        print(f"  {lab:16s} imaged & labelled {len(u):3d}   base rate {u[lab].mean():.3f}   "
              f"mean {ch} 1-vs-0 {d:+.3f}  (sign OK)")

    # Folds are assigned over the imaged, labelled rows only -- the rows a model can use.
    out = df[df["imaged"]].reset_index(drop=True)
    for lab, col in (("MEM", "MEM_decline_bi"), ("EF", "EF_decline_bi")):
        sub = out[out[col].notna()]
        f = assign_folds(sub, col, args.n_folds, args.seed, "group_assignment_T1")
        out.loc[sub.index, f"fold_id_{lab}_bi"] = f
        out[f"fold_id_{lab}_bi"] = out[f"fold_id_{lab}_bi"].astype("Int64")
        sizes = pd.Series(f).value_counts().sort_index().to_dict()
        print(f"  fold_id_{lab}_bi sizes {sizes}")

    cols = ["ID", "subject_id", "MEM_decline_bi", "EF_decline_bi", "delay_recall_ch",
            "exe_composite_ch", "em_pre", "em_post", "em_change",
            "ef_pre", "ef_post", "ef_change", "group_assignment_T1", "cognitive_status",
            "fold_id_MEM_bi", "fold_id_EF_bi"]
    out = out[[c for c in cols if c in out.columns]]
    if args.dry_run:
        print("\n--dry_run: nothing written")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} rows)  EM = {components}")


if __name__ == "__main__":
    main()
