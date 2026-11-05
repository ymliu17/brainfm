#!/usr/bin/env python3
"""One confirmatory table for the matched-task axis, replacing a 90-cell sweep.

WHY THIS EXISTS
---------------
Section 5c found the study's only coherent transfer -- an axis fitted on ADNI COGNITION,
not on the CN/Dementia diagnosis, predicts the RCT participants' BASELINE cognitive level
with the same sign in all three trials -- and immediately disqualified it: the sheet that
found it holds 90 correlated combined tests, and none would survive a family-wise
correction.

The task-matching principle fixes the multiplicity at its source rather than correcting for
it. If the ADNI fitting task must match the RCT downstream task, then the cell is DETERMINED
before any RCT number is read:

  * target      RCT BASELINE LEVEL, because that is the RCT endpoint being matched
  * construct   the same composite on both sides -- ADNI_MEM -> memory, ADNI_EF -> executive
  * direction   concurrent on both sides (scan and score at the same visit)
  * layer       `emb_fc` L3, fixed by the frozen ADNI sweep, which never saw an RCT scan
  * fit stage   MCI-only and all-stage-resid, the two arms sec 5c showed to tie in-source

That is 4 cells per encoder, not 90, and the choice of every one of them is forced by the
principle rather than by its result.

WHAT THIS SCRIPT CANNOT CLAIM
-----------------------------
The four cells were named AFTER the 90-cell sheet had been seen. Fixing them removes the
multiplicity of testing 90 things; it does NOT remove the look-elsewhere effect of having
looked. A real pre-registration needs data that does not exist yet. Every p below is
therefore reported as "conditional on a specification that was not blind", and the
counter-evidence that argues against the alignment mechanism is printed alongside rather
than in a footnote -- see the CROSSED block, where the strongest cells live.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAM = "BrainLM 13M emb_fc L3"

#: (fit arm, RCT target). Matched = the ADNI composite equals the RCT composite.
PRESPEC = [("MCI-MEM", "mem_level"), ("all-MEM resid", "mem_level"),
           ("MCI-EF", "ef_level"), ("all-EF resid", "ef_level")]
#: The same axes against the OTHER composite. Not pre-specified; printed because the
#: alignment hypothesis predicts these should be WEAKER, and they are not.
CROSSED = [("MCI-EF", "mem_level"), ("all-EF resid", "mem_level"),
           ("MCI-MEM", "ef_level"), ("all-MEM resid", "ef_level")]
#: Diagnosis positive class is Dementia. Negative raw correlations with cognition
#: have the expected direction; orient ONLY the summary, preserving raw correlations.
CONTROL = [("CN/AD", "mem_level"), ("CN/AD", "ef_level")]


def stouffer(cells):
    """Weighted Stouffer Z, the combination used throughout the study."""
    return sum(math.atanh(max(-0.999, min(0.999, r))) * math.sqrt(max(n - 3, 1))
               for r, n in cells) / math.sqrt(len(cells))


def two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run, m = [0.0] * len(ps), 0.0, len(ps)
    for k, i in enumerate(order):
        run = max(run, ps[i] * (m - k))
        out[i] = min(run, 1.0)
    return out


def cells_for(rows, arm, tgt):
    out = []
    for r in rows:
        if r["family"] != FAM or r["arm"] != arm:
            continue
        v, n = r.get(f"{tgt}_rho"), r.get(f"{tgt}_n")
        if v is not None and v == v and n:
            out.append((v, n, r["cohort"]))
    return out


def block(name, specs, rows, correct):
    """One block of cells. Holm is applied only where `correct` is set."""
    res = []
    for arm, tgt in specs:
        c = cells_for(rows, arm, tgt)
        if not c:
            continue
        direction = -1 if arm == "CN/AD" else 1
        raw_z = stouffer([(v, n) for v, n, _ in c])
        z = direction * raw_z
        res.append(dict(arm=arm, target=tgt, z=z, raw_z=raw_z,
                        expected_direction=direction, p=two_sided(z),
                        pos=sum(1 for v, _, _ in c if direction * v > 0), k=len(c),
                        per_cohort={co: round(v, 4) for v, _, co in c}))
    if correct and res:
        for r, h in zip(res, holm([r["p"] for r in res])):
            r["holm"] = h
    print(f"  {name}")
    for r in res:
        h = f"  holm {r['holm']:.3f}" if "holm" in r else ""
        print(f"    {r['arm']:16s} -> {r['target']:10s} Z {r['z']:+6.2f}  p {r['p']:.4f}"
              f"{h}   signs {r['pos']}/{r['k']}   {r['per_cohort']}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frozen", default="runs/v2/mci_axis_transfer_rep.json",
                    help="the frozen-encoder control. Must be the SAME script version as "
                         "the fine-tuned runs: mci_axis_transfer_5arm.json is not, and "
                         "differs from _rep in 11 of 27 shared cells because per-label "
                         "follow-up masking changed the sample.")
    ap.add_argument("--finetuned", nargs="*",
                    default=["runs/v2/mci_axis_transfer_ftEF.json",
                             "runs/v2/mci_axis_transfer_ftMEM.json"])
    ap.add_argument("--out", default="runs/benchmark_v1/confirm_matched_axis.json")
    a = ap.parse_args()

    runs = {"frozen": a.frozen}
    for f in a.finetuned:
        runs[Path(f).stem.replace("mci_axis_transfer_", "")] = f

    # Sample sizes first. A Z that rises because n rose is not a representation result.
    print("sample sizes per run (must be identical for the comparison to be like-for-like):")
    ns = {}
    for k, f in runs.items():
        rows = json.load(open(ROOT / f))["rows"]
        ns[k] = {(r["cohort"], t): r.get(f"{t}_n") for r in rows
                 if r["family"] == FAM and r["arm"] == "MCI-MEM"
                 for t in ("mem_level", "ef_level")}
        print(f"  {k:10s} {ns[k]}")
    ref = next(iter(ns.values()))
    same = all(v == ref for v in ns.values())
    print(f"  -> identical across runs: {same}")
    if not same:
        print("  WARNING: n differs; the Z comparison below is NOT like-for-like.")

    out = {"args": vars(a), "n_identical": bool(same), "runs": {},
           "status": "exploratory_sign_corrected_summary",
           "source_protocols": {k: json.load(open(ROOT / f)).get("protocol_version", "legacy") for k, f in runs.items()}}
    for k, f in runs.items():
        rows = json.load(open(ROOT / f))["rows"]
        print(f"\n=== {k} ===")
        out["runs"][k] = {
            "prespecified": block("EXPLORATORY matched construct, Holm over 4",
                                  PRESPEC, rows, correct=True),
            "crossed": block("CROSSED (counter-evidence, uncorrected)",
                             CROSSED, rows, correct=False),
            "control": block("DIAGNOSIS AXIS (expected raw direction: negative)",
                             CONTROL, rows, correct=False)}

    print(f"\n{'=' * 78}")
    fz = {(r["arm"], r["target"]): r["z"] for r in out["runs"]["frozen"]["prespecified"]}
    for k in out["runs"]:
        if k == "frozen":
            continue
        pre = out["runs"][k]["prespecified"]
        cro = out["runs"][k]["crossed"]
        surv = [r for r in pre if r.get("holm", 1) < 0.05]
        best_pre = max(pre, key=lambda r: r["z"])
        best_cro = max(cro, key=lambda r: r["z"])
        print(f"{k}: {len(surv)}/{len(pre)} selected matched cells survive Holm")
        for r in surv:
            print(f"    survivor: {r['arm']} -> {r['target']}  Z {r['z']:+.2f}  "
                  f"holm {r['holm']:.3f}  (frozen was {fz.get((r['arm'], r['target'])):+.2f})")
        print(f"    strongest matched {best_pre['z']:+.2f} "
              f"({best_pre['arm']} -> {best_pre['target']})")
        print(f"    strongest CROSSED       {best_cro['z']:+.2f} "
              f"({best_cro['arm']} -> {best_cro['target']})"
              + ("   <-- crossed exceeds matched; this descriptive comparison does not test the mechanism"
                 if best_cro["z"] > best_pre["z"] else ""))
    print("=" * 78)
    print("Every p above is conditional on a specification chosen after the 90-cell sheet")
    print("was seen. Four-cell Holm does not correct selection from the original 90-cell search.")

    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
