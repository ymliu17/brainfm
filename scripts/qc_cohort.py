#!/usr/bin/env python3
"""Quality control for the downstream ACT / CogTE dual-timepoint tensors.

Only the ADNI scans had been QC'd (scripts/qc_adni.py). The cohorts that carry the
paper's actual claims had not been checked at all.

Two things here that the ADNI pass could not ask:

* **T1 versus T2.** Each subject contributes a baseline and a post-intervention scan. If
  quality differs systematically between the two timepoints, the `diff` mode -- which
  subtracts the T1 embedding from the T2 embedding, and is the only downstream setting
  that beat its majority baseline -- inherits that difference as signal.
* **Quality versus outcome.** Whether responders and decliners differ in scan quality.
  A difference there is a confound in either direction: it can manufacture apparent
  prediction or bury real prediction.

Per-cohort acquisition is single-protocol (ACT TR 1.03 s, CogTE TR 2.5 s), so outliers are
judged within cohort x timepoint. Numbers are not comparable between the two cohorts:
DVARS and autocorrelation both scale with TR.

  python scripts/qc_cohort.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cohorts import load_cohort  # noqa: E402
from paths import COHORT, RUNS_DIR  # noqa: E402
from qc import add_flags, flag_rule_text, leave_one_out_typicality, scan_metrics  # noqa: E402
from temporal import COHORT_TR  # noqa: E402

REPORT = ["dvars_mean", "frac_spiky_frames", "max_abs", "gs_std", "lag1_ac",
          "frac_fc_gt09", "fc_typicality"]


def main():
    ap = argparse.ArgumentParser(description="QC the ACT / CogTE tensors")
    ap.add_argument("--cohorts", nargs="+", default=sorted(COHORT))
    ap.add_argument("--label", default="mem", choices=["mem", "ef"])
    ap.add_argument("--out", default=str(RUNS_DIR / "qc" / "cohort_qc.csv"))
    args = ap.parse_args()

    frames = []
    for cohort in args.cohorts:
        # require_labels=False: an unlabelled cohort (BREATHE) can still be QC'd.
        coh = load_cohort(cohort, require_labels=False)
        x = coh.bold.float().numpy()
        y = coh.labels(args.label).astype(int) if coh.table is not None else None
        ids = coh.subjects
        n, r, t, _ = x.shape
        tr = COHORT_TR.get(cohort)
        dur = f", TR {tr} s ({t * tr:.0f} s per scan)" if tr else ", TR not recorded"
        print(f"\n{cohort}: {n} subjects, {r} ROIs, {t} stored frames"
              f" ({int(coh.valid.min())}-{int(coh.valid.max())} usable){dur}", flush=True)

        rows, edges = [], []
        for tp_i, tp in enumerate(("T1", "T2")):
            for i in range(n):
                # Metrics are computed on the usable frames only: the tensors are
                # NaN-padded out to the cohort's longest scan, and DVARS / FC over a
                # padded tail is meaningless rather than merely noisy.
                m = scan_metrics(x[i, :, : int(coh.valid[i, tp_i]), tp_i])
                edges.append(m.pop("_edges"))
                m.update(cohort=cohort, timepoint=tp, subject=ids[i],
                         n_frames=int(coh.valid[i, tp_i]),
                         label=int(y[i]) if y is not None else -1)
                rows.append(m)
        df = pd.DataFrame(rows)
        # Typicality within cohort x timepoint: a T1 connectome should be judged against
        # other T1 connectomes, not against a pool that mixes the two sessions.
        E = np.vstack(edges)
        df["fc_typicality"] = np.nan
        for tp in ("T1", "T2"):
            sel = (df.timepoint == tp).values
            df.loc[sel, "fc_typicality"] = leave_one_out_typicality(E[sel])
        df["group"] = df["cohort"] + "_" + df["timepoint"]
        frames.append(add_flags(df, "group"))

    qc = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    qc.drop(columns=[c for c in qc.columns if c.startswith("rz_")]).to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("STRUCTURAL INTEGRITY")
    print("=" * 78)
    for g, sub in qc.groupby("group"):
        print(f"  {g:10s} n={len(sub):3d}  non-finite {int((sub.n_nonfinite > 0).sum())}"
              f"  dead ROIs {int((sub.n_constant_roi > 0).sum())}"
              f"  low-var ROIs {int((sub.n_lowvar_roi > 0).sum())}"
              f"  duplicate content {int(sub.sha1_8.duplicated(keep=False).sum())}")

    print("\n" + "=" * 78)
    print("T1 VERSUS T2  (median per timepoint; paired Wilcoxon over subjects)")
    print("=" * 78)
    for cohort in args.cohorts:
        sub = qc[qc.cohort == cohort]
        a = sub[sub.timepoint == "T1"].set_index("subject").sort_index()
        b = sub[sub.timepoint == "T2"].set_index("subject").sort_index()
        print(f"  {cohort}  (n={len(a)} subjects)")
        print(f"    {'metric':20s}{'T1':>10s}{'T2':>10s}{'delta':>10s}{'p':>10s}")
        for c in REPORT:
            p = stats.wilcoxon(a[c].values, b[c].values).pvalue
            star = " *" if p < 0.05 else ""
            print(f"    {c:20s}{a[c].median():10.4f}{b[c].median():10.4f}"
                  f"{b[c].median() - a[c].median():+10.4f}{p:10.3g}{star}")

    print("\n" + "=" * 78)
    print("IS QUALITY CONFOUNDED WITH THE OUTCOME?  (label 1 = responder, dEM >= 0)")
    print("=" * 78)
    # label == -1 marks a cohort with no outcome table; it has nothing to confound with.
    labelled = qc[qc.label >= 0]
    for g in sorted(set(qc.group) - set(labelled.group)):
        print(f"  {g}: no outcome table, skipped")
    for g, sub in labelled.groupby("group"):
        print(f"  {g}  (n={len(sub)}, responders {int(sub.label.sum())})")
        print(f"    {'metric':20s}{'resp':>10s}{'decl':>10s}{'AUC':>8s}{'p (MWU)':>10s}")
        for c in REPORT:
            r1, r0 = sub.loc[sub.label == 1, c], sub.loc[sub.label == 0, c]
            try:
                p = stats.mannwhitneyu(r1, r0).pvalue
                auc = roc_auc_score(sub.label, sub[c])
            except ValueError:
                p, auc = float("nan"), float("nan")
            star = " *" if p < 0.05 else ""
            print(f"    {c:20s}{r1.median():10.4f}{r0.median():10.4f}{auc:8.4f}{p:10.3g}{star}")

    print("\n" + "=" * 78)
    print(f"FLAGGED  (rule: {flag_rule_text()})")
    print("=" * 78)
    for g, sub in qc.groupby("group"):
        if (sub.label < 0).all():
            print(f"  {g:10s} flagged {int(sub.qc_flag.sum()):3d}/{len(sub):3d} "
                  f"({sub.qc_flag.mean():.1%})  no outcome table")
            continue
        fr = sub.groupby("label")["qc_flag"].mean()
        try:
            p = stats.chi2_contingency(pd.crosstab(sub.label, sub.qc_flag))[1]
        except ValueError:
            p = float("nan")
        print(f"  {g:10s} flagged {int(sub.qc_flag.sum()):3d}/{len(sub):3d} ({sub.qc_flag.mean():.1%})"
              f"  responder {fr.get(1, float('nan')):.3f} vs decliner {fr.get(0, float('nan')):.3f}"
              f"  (chi2 p={p:.3g})")

    # Subjects flagged at either timepoint are the ones a downstream run would drop.
    any_flag = qc.groupby(["cohort", "subject"])["qc_flag"].any().reset_index()
    print("\n  subjects flagged at either timepoint:")
    for cohort, sub in any_flag.groupby("cohort"):
        print(f"    {cohort}: {int(sub.qc_flag.sum())}/{len(sub)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
