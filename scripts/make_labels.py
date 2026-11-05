#!/usr/bin/env python3
"""Build a cohort's `labels.csv` -- binary outcomes plus stratified 5-fold assignments.

ACT and CogTE arrived with this file already built, by code that is not in the package.
BREATHE arrived as imaging only, so this script reconstructs the same schema from a raw
outcome table. It is also the tool to use when ACT's five unlabelled subjects get
outcomes, or when a cohort's folds have to be rebuilt.

What `labels.csv` must contain, because `cohorts.Cohort` and `paths.COHORT` read exactly
these names:

    ID                  subject, the join key against subjects.txt
    MEM_decline_bi      1 = responder (delta >= 0), 0 = decliner (delta < 0)
    EF_decline_bi       same encoding for executive function
    fold_id_MEM_bi      outer fold 1-5, stratified for the MEM outcome
    fold_id_EF_bi       outer fold 1-5, stratified for the EF outcome

Sign convention (the original data convention): despite the `decline` in the name, **1 is the
non-decline / responder class**. It is the positive class for F1 and sensitivity.

Stratification follows what the shipped files did. ACT's folds are stratified on the
outcome alone; CogTE's are stratified on trial arm x outcome, and the arm-balanced form is
the better default -- an arm that lands unevenly across folds turns the intervention into
a fold-level confound. Pass `--arm_col` to get it.

Two inputs are needed per outcome: the T1 and T2 scores. The binary label is
`(T2 - T1) >= 0`, computed here rather than taken from the table, so the sign convention
is applied in one place. Subjects missing either score are dropped and named.

Examples:
  # BREATHE, once an outcome table exists
  python scripts/make_labels.py --cohort breathe --table raw_breathe_cog.csv \
      --id_col record_id --arm_col group \
      --mem_t1 delay_recall_T1 --mem_t2 delay_recall_T2 \
      --ef_t1 exec_composite_T1 --ef_t2 exec_composite_T2

  # check what would be written without writing it
  python scripts/make_labels.py --cohort breathe --table raw.csv ... --dry_run
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

N_FOLDS = 5


def assign_folds(strata: pd.Series, seed: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """Deal each stratum round-robin across folds, so every fold mirrors the whole.

    Round-robin within a stratum rather than a random split: with 15-20 subjects per fold
    a random assignment routinely leaves a fold with zero minority-class members, which
    makes its AUC undefined. Dealing guarantees the closest achievable balance. The seed
    only shuffles the order within a stratum, so the counts per fold are deterministic.
    """
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(strata), dtype=int)
    for _, idx in strata.groupby(strata, sort=True).groups.items():
        pos = np.asarray(idx)
        pos = pos[rng.permutation(len(pos))]
        for i, p in enumerate(pos):
            fold[strata.index.get_loc(p)] = i % n_folds + 1
    return fold


def main():
    ap = argparse.ArgumentParser(description="Build a cohort labels.csv with folds")
    ap.add_argument("--cohort", choices=sorted(COHORT), required=True)
    ap.add_argument("--table", required=True, help="raw outcome table (CSV)")
    ap.add_argument("--id_col", default="ID", help="subject column in --table")
    ap.add_argument("--mem_t1", required=True, help="episodic-memory score at T1")
    ap.add_argument("--mem_t2", required=True, help="episodic-memory score at T2")
    ap.add_argument("--ef_t1", default=None, help="executive-function score at T1")
    ap.add_argument("--ef_t2", default=None, help="executive-function score at T2")
    ap.add_argument("--arm_col", default=None,
                    help="trial-arm column; folds are then stratified on arm x outcome, "
                         "as CogTE's shipped folds are")
    ap.add_argument("--id_prefix", default=None,
                    help="prepend this to --id_col values, e.g. 'sub-' if the outcome "
                         "table stores bare numeric IDs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="default: data/rct/<cohort>/labels.csv")
    ap.add_argument("--dry_run", action="store_true", help="report, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing labels.csv")
    args = ap.parse_args()

    meta = COHORT[args.cohort]
    out = Path(args.out) if args.out else Path(meta["labels_csv"])
    if out.exists() and not (args.force or args.dry_run):
        raise SystemExit(f"{out} exists; pass --force to replace it (the old folds are "
                         "part of every published number, so this is deliberate)")

    subjects = [s for s in Path(meta["subjects"]).read_text().split() if s]
    raw = pd.read_csv(args.table)
    if args.id_col not in raw.columns:
        raise SystemExit(f"{args.table} has no {args.id_col!r} column; it has "
                         f"{list(raw.columns)[:20]}")
    raw = raw.rename(columns={args.id_col: "ID"})
    raw["ID"] = raw["ID"].astype(str)
    if args.id_prefix:
        raw["ID"] = args.id_prefix + raw["ID"]
    if raw["ID"].duplicated().any():
        dup = sorted(raw.loc[raw["ID"].duplicated(), "ID"])
        raise SystemExit(f"{args.table} has {len(dup)} duplicate IDs, e.g. {dup[:5]}. "
                         "One row per subject is required -- reshape to wide first.")

    outcomes = {"MEM": (args.mem_t1, args.mem_t2)}
    if args.ef_t1 and args.ef_t2:
        outcomes["EF"] = (args.ef_t1, args.ef_t2)
    missing = [c for pair in outcomes.values() for c in pair if c not in raw.columns]
    if missing:
        raise SystemExit(f"{args.table} is missing score columns: {missing}")

    df = raw[["ID"] + ([args.arm_col] if args.arm_col else [])].copy()
    for name, (t1, t2) in outcomes.items():
        delta = pd.to_numeric(raw[t2], errors="coerce") - pd.to_numeric(raw[t1], errors="coerce")
        df[f"{name}_delta"] = delta
        df[f"{name}_decline_bi"] = (delta >= 0).astype("Int64").where(delta.notna())

    # Only subjects that have imaging AND every requested outcome can carry a fold.
    imaged = df["ID"].isin(subjects)
    complete = df[[f"{n}_decline_bi" for n in outcomes]].notna().all(axis=1)
    print(f"outcome table: {len(df)} rows")
    print(f"  {int(imaged.sum())} have imaging in data/rct/{args.cohort}/ "
          f"({len(subjects)} scanned)")
    dropped_score = sorted(df.loc[imaged & ~complete, "ID"])
    if dropped_score:
        print(f"  {len(dropped_score)} dropped for a missing score: {dropped_score}")
    no_outcome = sorted(set(subjects) - set(df.loc[complete, "ID"]))
    if no_outcome:
        print(f"  {len(no_outcome)} imaged subjects have no usable outcome row and will "
              f"be excluded at load time: {no_outcome}")
    df = df[imaged & complete].reset_index(drop=True)
    if df.empty:
        raise SystemExit("no subject has both imaging and a complete outcome -- nothing to write")

    for name in outcomes:
        label = df[f"{name}_decline_bi"].astype(int)
        strat = label.astype(str)
        if args.arm_col:
            strat = df[args.arm_col].astype(str) + "_" + strat
        strat.name = f"stratify_label_{name}_bi"
        df[strat.name] = strat
        df[f"fold_id_{name}_bi"] = assign_folds(strat, args.seed)
        print(f"\n{name}: n={len(df)} balance {label.value_counts().sort_index().to_dict()} "
              f"majority={label.value_counts(normalize=True).max():.4f}")
        print(pd.crosstab(strat, df[f"fold_id_{name}_bi"]).to_string())
        counts = df[f"fold_id_{name}_bi"].value_counts().sort_index()
        if counts.min() < 5:
            print(f"  WARNING: smallest fold has {counts.min()} subjects")
        per_fold_min = df.groupby(f"fold_id_{name}_bi")[f"{name}_decline_bi"].apply(
            lambda v: min((v == 0).sum(), (v == 1).sum()))
        if (per_fold_min == 0).any():
            print(f"  WARNING: folds {list(per_fold_min[per_fold_min == 0].index)} contain "
                  "only one class, so their AUC is undefined")

    df = df.drop(columns=[c for c in df.columns if c.endswith("_delta")])
    if args.dry_run:
        print(f"\n--dry_run: would write {len(df)} rows x {len(df.columns)} cols to {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows: {list(df.columns)})")
    print(f"{args.cohort} is now a labelled cohort; scripts/finetune_decline.py "
          f"--cohort {args.cohort} will accept it.")


if __name__ == "__main__":
    main()
