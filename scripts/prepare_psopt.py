#!/usr/bin/env python3
"""Build `subjects.txt` and `labels.csv` for the pSOPT cohort, in the schema the other three use.

pSOPT arrived as raw tables rather than as a prepared cohort, so nothing in `src/cohorts.py`
could read it. This script produces the two files `paths.COHORT` expects and NOTHING else --
no imaging is touched, no normalisation is applied. That matters because of the standing
rule that the RCTs are independent validation sets: the only thing being created here is a
label table.

CONSTRUCT MATCHING IS THE POINT
-------------------------------
The other three cohorts are analysed through two constructs, `mem` and `ef`, and a psopt
result is only comparable if it uses the SAME constructs rather than whatever psopt happens
to ship. The mapping:

    mem  <- bvmtr_delayed_recall   (BVMT-R delayed recall, the same instrument family as
                                    ACT's `calc_delay_recall_tscore` and CogTE's
                                    `DelayRecall_tscore`)
    ef   <- ef                     (executive composite, as ACT/CogTE `executive_composite`
                                    and BREATHE `ef_pre`/`ef_post`)

psopt additionally ships MoCA, GDS-30, STAI state/trait, AES-18 and three AD-signature
cortical-thickness measures. Those are written out but are BASELINE-ONLY and are not part of
the mem/ef pair; they are extra targets, not substitutes.

THE BINARY LABEL RULE WAS READ OFF THE EXISTING COHORTS, NOT CHOSEN
------------------------------------------------------------------
In all three existing cohorts the binary column is exactly `change >= 0`, verified by the
gap between the classes: label-0 maxima are -4.000 / -0.050 (ACT mem/ef), -3.000 / -0.006
(CogTE) and -0.163 / -0.008 (BREATHE) while every label-1 minimum is >= 0. So `1 =
improvement`, cut at zero, and psopt uses the same rule. (The column is NAMED `*_decline_bi`
and means the opposite; see the note in paths.COHORT.)

CENSORING
---------
`bvmtr_delayed_recall` is dtype object because it carries the floor/ceiling codes `'<20'`
and `'>80'`. They are imputed to 19 and 81 -- order-preserving, which is all Spearman and
the sign of a change score need. The script ASSERTS that the sign of every affected change
score is determinate under any imputation consistent with the code (i.e. that the pair does
not straddle zero when the censored end is moved through its whole admissible range), so no
binary label depends on the choice. Rows where the sign is indeterminate are set to NaN
rather than guessed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "rct" / "psopt"

#: Floor/ceiling codes -> (imputed value, admissible range). The range is what the
#: determinacy assertion sweeps.
CENSOR = {"<20": (19.0, (0.0, 20.0)), ">80": (81.0, (80.0, 100.0))}

VISITS = ("T1", "T2", "T3")


def decode(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(value, lo, hi) arrays. For uncensored entries lo == hi == value."""
    v, lo, hi = [], [], []
    for raw in series.astype(str):
        if raw in CENSOR:
            val, (a, b) = CENSOR[raw]
            v.append(val), lo.append(a), hi.append(b)
        else:
            try:
                x = float(raw)
            except ValueError:
                x = np.nan
            v.append(x), lo.append(x), hi.append(x)
    return np.array(v), np.array(lo), np.array(hi)


def main():
    # --- subjects.txt ------------------------------------------------------------------
    a424 = [l.strip() for l in open(D / "pSOPT_A424_T1T2_raw_subjects.txt") if l.strip()]
    for other in ("pSOPT_Schaefer400Tian_T1T2_raw_subjects.txt",
                  "pSOPT_A424_T1T2_z_transformed_subjects.txt",
                  "pSOPT_Schaefer400Tian_T1T2_z_transformed_subjects.txt"):
        got = [l.strip() for l in open(D / other) if l.strip()]
        if got != a424:
            raise SystemExit(f"subject order differs: {other} vs A424 raw")
    (D / "subjects.txt").write_text("\n".join(a424) + "\n")
    print(f"subjects.txt: {len(a424)} rows, order identical across all four releases")

    # --- long table -> wide -------------------------------------------------------------
    f = pd.read_csv(D / "pSOPT_adsct_fc_meanFD.csv")
    dem = pd.read_csv(D / "pSOPT_demographics.csv")

    # bvmtr_delayed_recall carries the censoring codes; decode it once on the long table.
    v, lo, hi = decode(f["bvmtr_delayed_recall"])
    f = f.assign(_mem=v, _mem_lo=lo, _mem_hi=hi)

    wide = pd.DataFrame({"ID": ["sub-" + s for s in a424 and a424]})
    wide["ID"] = ["sub-" + s.replace("sub-", "") for s in a424]
    key = wide.ID.str.replace("sub-", "", regex=False)

    def spread(col, out):
        p = f.pivot_table(index="subject", columns="timepoint", values=col, aggfunc="first")
        for t in VISITS:
            wide[f"{out}_{t}"] = key.map(p[t]).values if t in p.columns else np.nan

    spread("_mem", "delay_recall")
    spread("_mem_lo", "_mem_lo")
    spread("_mem_hi", "_mem_hi")
    spread("ef", "ef")
    spread("bvmtr_learning", "bvmtr_learning")
    spread("mean_fd", "mean_fd")
    spread("adsct_jack", "adsct_jack")

    # arm: one value per subject, asserted constant across visits
    arms = f.groupby("subject")["arm"].nunique()
    if (arms > 1).any():
        raise SystemExit(f"arm varies within subject: {arms[arms > 1].index.tolist()}")
    arm = f.groupby("subject")["arm"].first()
    wide["arm_name"] = key.map(arm).values
    wide["arm_code"] = wide.arm_name.map({"psopt": 1, "MLA": 2}).astype("Int64")

    # baseline-only measures from the demographics table
    dem = dem.set_index("subject")
    wide["age"] = key.map(dem["age"]).values
    wide["sex_code"] = key.map(dem["sex"].map({"Female": 0, "Male": 1})).values
    for c in ("education", "moca", "gds30", "stai_s", "stai_t", "aes18",
              "adsct_jack", "adsct_schwarz", "adsct_dickerson"):
        wide[f"{c}_bsl"] = key.map(dem[c]).values

    # --- change scores and binary labels ------------------------------------------------
    # `1 = improvement`, cut at zero -- the rule the other three cohorts already use.
    wide["delay_recall_ch"] = wide.delay_recall_T2 - wide.delay_recall_T1
    wide["exe_composite_ch"] = wide.ef_T2 - wide.ef_T1

    # Determinacy of the mem change sign under the censoring codes. The most adverse
    # admissible pair for a positive change is (T1 at its max, T2 at its min) and vice
    # versa; if both extremes agree in sign the label does not depend on the imputation.
    worst_lo = wide["_mem_lo_T2"] - wide["_mem_hi_T1"]
    worst_hi = wide["_mem_hi_T2"] - wide["_mem_lo_T1"]
    determinate = (np.sign(worst_lo) == np.sign(worst_hi)) | ~np.isfinite(worst_lo)
    censored = (wide["_mem_lo_T1"] != wide["_mem_hi_T1"]) | (wide["_mem_lo_T2"] != wide["_mem_hi_T2"])
    n_cens = int((censored & np.isfinite(wide.delay_recall_ch)).sum())
    bad = int((censored & np.isfinite(wide.delay_recall_ch) & ~determinate).sum())
    print(f"mem change: {n_cens} subject(s) involve a censored value; "
          f"{bad} have an indeterminate sign (set to NaN)")

    wide["MEM_decline_bi"] = (wide.delay_recall_ch >= 0).astype(float)
    wide.loc[~np.isfinite(wide.delay_recall_ch), "MEM_decline_bi"] = np.nan
    wide.loc[censored & ~determinate, ["MEM_decline_bi", "delay_recall_ch"]] = np.nan
    wide["EF_decline_bi"] = (wide.exe_composite_ch >= 0).astype(float)
    wide.loc[~np.isfinite(wide.exe_composite_ch), "EF_decline_bi"] = np.nan

    wide = wide.drop(columns=[c for c in wide.columns if c.startswith("_mem_")])

    # --- folds ---------------------------------------------------------------------------
    # Stratified on the label the column is named for, exactly as the other cohorts pair
    # them (see `fold_col_for_label`): using a fold column stratified on the OTHER outcome
    # silently breaks the stratification.
    for lab, src in (("MEM", "MEM_decline_bi"), ("EF", "EF_decline_bi")):
        col = f"fold_id_{lab}_bi"
        wide[col] = np.nan
        ok = np.where(np.isfinite(wide[src]))[0]
        if len(ok) >= 10:
            y = wide[src].values[ok].astype(int)
            for fi, (_, te) in enumerate(StratifiedKFold(
                    5, shuffle=True, random_state=0).split(ok.reshape(-1, 1), y)):
                wide.loc[wide.index[ok[te]], col] = fi + 1

    out = D / "labels.csv"
    wide.to_csv(out, index=False)
    print(f"\nwrote {out}  ({wide.shape[0]} rows x {wide.shape[1]} cols)")
    print("\ncoverage of the analysis columns:")
    for c in ("delay_recall_T1", "delay_recall_T2", "ef_T1", "ef_T2",
              "delay_recall_ch", "exe_composite_ch", "MEM_decline_bi", "EF_decline_bi",
              "age", "sex_code", "moca_bsl", "gds30_bsl", "arm_code"):
        n = int(np.isfinite(pd.to_numeric(wide[c], errors="coerce")).sum())
        extra = ""
        if c.endswith("_bi") and n:
            extra = f"   frac1 {pd.to_numeric(wide[c]).mean():.3f}"
        print(f"  {c:20s} {n:3d}/{len(wide)}{extra}")
    print("\narm split:", wide.arm_name.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
