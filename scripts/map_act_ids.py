#!/usr/bin/env python3
"""Recover the ACT imaging-ID <-> analytical-CSV-ID mapping, and verify it.

WHY THIS IS NEEDED
------------------
`data/rct/act/act_cog_analytical.csv` (2026-08-24) carries the covariates the revision
needs -- study_site, group_assignment, peak VO2, session attendance, MoCA, GDS, education,
BMI, comorbidity count -- but keys them by `formx_subj_study_id`, an integer in 1..8033.
The imaging (`subjects.txt`) and the shipped `labels.csv` key by `sub-0105`-style IDs.
The two ID spaces have ZERO overlap, so no string transform joins them, and no mapping
file was supplied.

HOW THE MAPPING IS RECOVERED
----------------------------
`labels.csv` contains both the imaging ID and the baseline cognitive scores. Those scores
also appear in the analytical CSV under the integer ID. Matching on the *pair*
(baseline delay-recall t-score, baseline executive composite) identifies each subject
uniquely -- the pair is effectively a fingerprint.

A fingerprint match is only trustworthy if it predicts things it was not fitted on, so the
recovered mapping is checked against two columns that take no part in the matching:

    6-month delay-recall t-score   (labels.csv T2 vs the CSV's 6m_assessment_arm_1)
    group_assignment               (labels.csv vs the CSV's baseline row)

Both must agree for every subject or the script refuses to write anything. Measured on the
2026-08-24 files: 95/95 uniquely matched, 95/95 agreement on each check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paths import COHORT, DERIVED_DIR  # noqa: E402

ANALYTICAL = COHORT["act"]["dir"] / "act_cog_analytical.csv"
BASELINE_EVENT = "baseline_arm_1"
FOLLOWUP_EVENT = "6m_assessment_arm_1"       # the event labels.csv's *_T2 columns come from

#: Covariates worth carrying downstream. Everything here is measured at baseline, i.e.
#: before the outcome, so conditioning on it cannot leak the outcome.
COVARIATES = [
    "study_site", "group_assignment", "formj_subject_age", "formj_subj_gender",
    "formj_subj_race", "formj_subj_ethnicity", "formj_subj_education_years", "bmi",
    "calc_moca_total", "forme_wtar_score", "calc_gds_total", "formm_peak_vo2",
    "formb_on_mem_meds", "formb_on_psy_meds", "calc_formg_adl_total", "num_comorbidity",
    "formx_total_session_attend", "formx_hrpas", "formj_subj_marital_status",
    "formj_subj_income",
]


def main():
    ap = argparse.ArgumentParser(description="Recover and verify the ACT ID mapping")
    ap.add_argument("--out_dir", default=str(DERIVED_DIR / "rct"))
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    labels = pd.read_csv(COHORT["act"]["labels_csv"])
    ana = pd.read_csv(ANALYTICAL, low_memory=False)
    base = ana[ana["redcap_event_name"] == BASELINE_EVENT].set_index("formx_subj_study_id")
    foll = ana[ana["redcap_event_name"] == FOLLOWUP_EVENT].set_index("formx_subj_study_id")
    if base.index.duplicated().any():
        raise SystemExit(f"{ANALYTICAL.name}: duplicate subjects in {BASELINE_EVENT}")

    mapping, ambiguous, unmatched = {}, [], []
    for r in labels.itertuples():
        hit = base.index[
            np.isclose(base["calc_delay_recall_tscore"], r.calc_delay_recall_tscore_T1,
                       atol=args.tol)
            & np.isclose(base["executive_composite"], r.executive_composite_T1, atol=args.tol)
        ]
        if len(hit) == 1:
            mapping[r.ID] = int(hit[0])
        elif len(hit) == 0:
            unmatched.append(r.ID)
        else:
            ambiguous.append((r.ID, [int(h) for h in hit]))

    print(f"{len(labels)} labelled subjects: matched {len(mapping)}, "
          f"ambiguous {len(ambiguous)}, unmatched {len(unmatched)}")
    if ambiguous or unmatched:
        raise SystemExit(f"refusing to write a partial mapping. ambiguous={ambiguous[:5]} "
                         f"unmatched={unmatched[:5]}")

    # --- held-out checks: columns the fingerprint did not use -----------------------
    checks = {}
    ok = bad = 0
    for sid, fid in mapping.items():
        want = float(labels.loc[labels["ID"] == sid, "calc_delay_recall_tscore_T2"].iloc[0])
        if fid not in foll.index or not np.isfinite(want):
            continue
        got = float(foll.loc[fid, "calc_delay_recall_tscore"])
        ok, bad = (ok + 1, bad) if np.isclose(got, want, atol=args.tol) else (ok, bad + 1)
    checks["delay_recall_T2"] = (ok, bad)

    ok = bad = 0
    for sid, fid in mapping.items():
        want = labels.loc[labels["ID"] == sid, "group_assignment_T1"].iloc[0]
        got = base.loc[fid, "group_assignment"]
        ok, bad = (ok + 1, bad) if got == want else (ok, bad + 1)
    checks["group_assignment"] = (ok, bad)

    for name, (a, b) in checks.items():
        print(f"  held-out check {name}: {a} agree / {b} disagree")
    if any(b > 0 for _, b in checks.values()):
        raise SystemExit("a held-out check failed: the fingerprint match is not the true "
                         "mapping. Nothing written.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "act_id_map.json").write_text(json.dumps(mapping, indent=1, sort_keys=True))

    cov = base.reindex([mapping[s] for s in labels["ID"]])
    keep = [c for c in COVARIATES if c in cov.columns]
    tbl = cov[keep].reset_index(drop=True)
    tbl.insert(0, "ID", labels["ID"].values)
    tbl.to_csv(out_dir / "act_covariates.csv", index=False)
    missing = [c for c in COVARIATES if c not in cov.columns]
    print(f"\nwrote {out_dir}/act_id_map.json ({len(mapping)} subjects)")
    print(f"wrote {out_dir}/act_covariates.csv ({len(keep)} covariates)"
          + (f"; absent from the CSV: {missing}" if missing else ""))
    print(f"  study_site: {tbl['study_site'].value_counts().to_dict()}"
          if "study_site" in tbl else "")
    nn = tbl[keep].notna().mean().sort_values()
    print("  least-complete covariates: "
          + ", ".join(f"{c} {v * 100:.0f}%" for c, v in nn.head(4).items()))


if __name__ == "__main__":
    main()
