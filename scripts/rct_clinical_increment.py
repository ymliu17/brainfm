#!/usr/bin/env python3
"""Baseline-only prediction of follow-up cognition: clinical and imaging increments.

Nested preprocessing is fitted within the corresponding training split. Paired
bootstrap intervals condition on the fitted OOF predictions, not on refitting the
whole model. MEM/EF baseline adjustment does not by itself reduce noise compared
with an equivalent baseline-adjusted change model. Ridge penalties and rank-based
scoring must be interpreted on their actual target scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from benchmark_evaluation import PROTOCOL_VERSION
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from cohorts import center_rois, load_cohort  # noqa: E402
from datasets import load_xyz  # noqa: E402
from fusion_increment import oof_predictions, paired_bootstrap, score  # noqa: E402
from mci_axis_transfer import Axis, adni_features  # noqa: E402
from paths import DERIVED_DIR, OFFICIAL_BRAINLM_13M, XYZ_PATH  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Encoder arms. `frozen` is the official checkpoint; the other two are §5i's interpolated
#: points -- a=1 is the encoder actually produced by post-training, a=2 the extrapolated
#: peak of the transfer dose-response. Both are reported because a=1 is what exists and a=2
#: is the best case the scan found.
ENCODERS = [("FM", str(OFFICIAL_BRAINLM_13M)),
            ("FMft1", "checkpoints/alpha/brainlm13m_adniMEM_level_a1.pt"),
            ("FMft2", "checkpoints/alpha/brainlm13m_adniMEM_level_a2.pt")]

#: ADNI source for the transferred axis, per RCT composite: the matched pairing of §5h.
AXIS_SRC = {"mem": "raw_ADNI_MEM_all_t200_off_dxMCI",
            "ef": "raw_ADNI_EF_all_t200_off_dxMCI"}


#: DO NOT combine these deltas with the study's `stouffer()`. That function computes
#: `atanh(r) * sqrt(n-3)`, whose standard error is the one for a RAW correlation. Every
#: number here is a PAIRED DIFFERENCE of two correlations on the same subjects, and because
#: the arms share subjects -- often differing only in an encoder whose weights moved by 5
#: parts in ten thousand -- the difference has a far smaller standard error than a single
#: correlation does. Using the Fisher form understated one cell by 4x (breathe/ef,
#: clin+FMft1ax vs clin+FMax: bootstrap z +2.84, Fisher form +0.74). The paired bootstrap
#: already produces the right standard error, so combination is inverse-variance meta over
#: `delta / se_bootstrap`. See scripts/rct_clinical_meta.py, which is where combination now
#: happens; this script writes per-cohort deltas, CIs and bootstrap p only.


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run, m = [0.0] * len(ps), 0.0, len(ps)
    for k, i in enumerate(order):
        run = max(run, ps[i] * (m - k))
        out[i] = min(run, 1.0)
    return out


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mci_dir", default=str(DERIVED_DIR / "adni424_t200_mci"))
    ap.add_argument("--tlen", type=int, default=200)
    ap.add_argument("--cohorts", nargs="+", default=["act", "cogte", "breathe"])
    ap.add_argument("--labels", nargs="+", default=["mem", "ef"])
    ap.add_argument("--k", type=int, default=5,
                    help="components per imaging block in the `pca` arm. Small on purpose: "
                         "a training fold here is 59-86 subjects, so clin+FC(2k) at k=5 is "
                         "already 14 columns.")
    ap.add_argument("--n_pca_axis", type=int, default=200,
                    help="components for the ADNI-side axis fit (matches mci_axis_transfer)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--release", default="amp", choices=("amp", "zscore", "schaefer", "amp_nogsr", "zscore_nogsr", "schaefer_nogsr"),
                    help="Which stored file to read. The *_nogsr variants exist only for psopt, "
                         "which ships the same recordings under two denoisings "
                         "(pSOPT_visits12_GSR / pSOPT_visits12_noGSR, delivered "
                         "2026-09-17). Default `amp` is the GSR build, matching what "
                         "act/cogte/breathe's A424 releases are.")
    ap.add_argument("--out", default="runs/benchmark_v1/rct_clinical_increment.json")
    a = ap.parse_args()
    C.configure_backends()
    C.NUM_TIMEPOINTS_PER_VOXEL = a.tlen

    from model_loader import load_brainlm_forpretraining
    from rct_positive_control import CONTROLS
    from rct_probe import upsample_short

    # --- cohorts, T1 only ------------------------------------------------------------
    coh = {}
    for n in a.cohorts:
        co = center_rois(load_cohort(n, release=a.release, require_labels=False))
        if int(co.valid.min()) < a.tlen:
            co, _ = upsample_short(co, a.tlen, C.TIMEPOINT_PATCHING_SIZE, fill="pad")
        coh[n] = co
    xyz = load_xyz(XYZ_PATH)

    # --- ADNI tensors for the axis fits ----------------------------------------------
    adni = {}
    for lb, tag in AXIS_SRC.items():
        if lb not in a.labels:
            continue
        x = torch.load(f"{a.mci_dir}/ADNI_x_{tag}.pt", map_location="cpu",
                       weights_only=False).float()
        y = torch.load(f"{a.mci_dir}/ADNI_y_{tag}.pt", map_location="cpu",
                       weights_only=False).numpy().astype(float)
        adni[lb] = (x, y)

    # --- features: FC once, each encoder once ----------------------------------------
    print("building features", flush=True)
    RF, AF = {}, {}                              # RF[(name, cohort)], AF[(name, label)]
    for n, co in coh.items():
        RF[("FC", n)] = adni_features("fc", co.bold[..., 0][:, :, :a.tlen],
                                      None, xyz, None, None, 64, None)
    for lb, (x, _) in adni.items():
        AF[("FC", lb)] = adni_features("fc", x[:, :, :a.tlen], None, xyz,
                                       None, None, 64, None)
    gpu_acquire()
    for name, ck in ENCODERS:
        model = load_brainlm_forpretraining(Path(ck)).to(DEVICE).eval()
        for n, co in coh.items():
            RF[(name, n)] = adni_features("brainlm", co.bold[..., 0][:, :, :a.tlen],
                                          model, xyz, "emb_fc", 3, 64, None)
        for lb, (x, _) in adni.items():
            AF[(name, lb)] = adni_features("brainlm", x[:, :, :a.tlen], model, xyz,
                                           "emb_fc", 3, 64, None)
        del model
        torch.cuda.empty_cache()
    from benchmark_gpu_phase import release_gpu_for_cpu_tail
    release_gpu_for_cpu_tail()
    fams = ["FC"] + [nm for nm, _ in ENCODERS]

    # --- ADNI-fitted axes, applied to the RCT ----------------------------------------
    AX = {}
    for lb in adni:
        _, y = adni[lb]
        for f in fams:
            ax = Axis(a.n_pca_axis, False).fit(AF[(f, lb)], y, None)
            for n in coh:
                AX[(f, lb, n)] = ax.score(RF[(f, n)])

    rows, tests, predictions = [], [], []
    for lb in a.labels:
        for n, co in coh.items():
            ctrl = CONTROLS[n](co.subjects)
            yv = np.asarray(co.followup_levels(lb), float)
            base = co.labelled_mask(lb)
            clin_cols = [np.asarray(co.baseline_levels(l2), float) for l2 in ("mem", "ef")]
            clin_cols += [np.asarray(ctrl["age"][0], float),
                          np.asarray(ctrl["sex"][0], float)]
            m = base & np.isfinite(yv)
            for c in clin_cols:
                m &= np.isfinite(c)
            if int(m.sum()) < 30:
                print(f"  {n}/{lb}: n={int(m.sum())} after complete-case, skipped",
                      flush=True)
                continue
            y = yv[m]
            clin = np.column_stack([c[m] for c in clin_cols])
            print(f"\n=== {n} / {lb}_t2   n={len(y)} complete cases ===", flush=True)

            # --- arms -----------------------------------------------------------------
            #: (arm name, list of blocks, list of k per block). The clinical block is never
            #: reduced: it is 4 interpretable columns and PCA on 4 columns buys nothing.
            arms = {"clin": ([clin], [4])}
            rng = np.random.default_rng(0)
            arms["clin+perm"] = ([clin, rng.permutation(
                AX[("FC", lb, n)][m]).reshape(-1, 1)], [4, 1])
            for f in fams:
                arms[f"clin+{f}ax"] = ([clin, AX[(f, lb, n)][m].reshape(-1, 1)], [4, 1])
            arms["clin+FCax+FMax"] = ([clin, AX[("FC", lb, n)][m].reshape(-1, 1),
                                       AX[("FM", lb, n)][m].reshape(-1, 1)], [4, 1, 1])
            k = a.k
            arms[f"clin+FC({k})"] = ([clin, RF[("FC", n)][m]], [4, k])
            arms[f"clin+FC({2 * k})"] = ([clin, RF[("FC", n)][m]], [4, 2 * k])
            for f in [nm for nm, _ in ENCODERS]:
                arms[f"clin+{f}({k})"] = ([clin, RF[(f, n)][m]], [4, k])
            arms[f"clin+FM({2 * k})"] = ([clin, RF[("FM", n)][m]], [4, 2 * k])
            arms[f"clin+FC({k})+FM({k})"] = ([clin, RF[("FC", n)][m], RF[("FM", n)][m]],
                                             [4, k, k])
            arms[f"clin+FC({k})+FMft1({k})"] = ([clin, RF[("FC", n)][m],
                                                 RF[("FMft1", n)][m]], [4, k, k])

            oof = {}
            for nm, (blocks, ks) in arms.items():
                p = np.mean([oof_predictions(blocks, ks, y, s) for s in a.seeds], axis=0)
                oof[nm] = p
                sc = score(y, p)
                rows.append(dict(cohort=n, label=lb, arm=nm, n=int(len(y)), score=sc,
                                 mae=float(mean_absolute_error(y, p)),
                                 rmse=float(np.sqrt(mean_squared_error(y, p))),
                                 r2=float(r2_score(y, p))))
                print(f"  {nm:<26} rho {sc:+.4f}", flush=True)

            predictions.append({"cohort": n, "label": lb,
                                "subjects": co.subjects[m].astype(str).tolist(),
                                "y": y.tolist(),
                                "oof": {nm: p.tolist() for nm, p in oof.items()}})
            #: (a, b, block, note). `b` earns credit over `a`. `block` groups the tests for
            #: Holm. Every k-dim increment is stated against its dimension-matched control
            #: as well, never against the narrower arm alone.
            PAIRS = [
                ("clin", "clin+FCax", "axis", "does FC add over clinical?"),
                ("clin", "clin+FMax", "axis", "does the frozen FM add over clinical?"),
                ("clin", "clin+FMft1ax", "axis", "does the adapted FM add over clinical?"),
                ("clin", "clin+FMft2ax", "axis", "same, at the extrapolated peak"),
                ("clin+perm", "clin+FMax", "axis", "vs one arbitrary extra column"),
                ("clin+perm", "clin+FCax", "axis", "vs one arbitrary extra column"),
                ("clin+FMax", "clin+FMft1ax", "adapt", "did post-training buy anything?"),
                ("clin+FMax", "clin+FMft2ax", "adapt", "same, at the peak"),
                ("clin+FCax", "clin+FCax+FMax", "axis", "FM on top of FC, both 1-dim"),
                ("clin", f"clin+FC({k})", "pca", "FC block over clinical"),
                ("clin", f"clin+FM({k})", "pca", "frozen FM block over clinical"),
                ("clin", f"clin+FMft1({k})", "pca", "adapted FM block over clinical"),
                (f"clin+FM({k})", f"clin+FMft1({k})", "adapt",
                 "post-training, matched dimension"),
                (f"clin+FC({k})", f"clin+FC({k})+FM({k})", "pca",
                 "NAIVE control -- read the next line before this one"),
                (f"clin+FC({2 * k})", f"clin+FC({k})+FM({k})", "pca",
                 "DIMENSION-MATCHED control"),
                (f"clin+FC({2 * k})", f"clin+FC({k})+FMft1({k})", "pca",
                 "DIMENSION-MATCHED, adapted"),
                (f"clin+FM({2 * k})", f"clin+FC({k})+FM({k})", "pca",
                 "dimension-matched the other way"),
            ]
            for u, v, blk, note in PAIRS:
                if u not in oof or v not in oof:
                    continue
                d, p, ci, detail = paired_bootstrap(y, oof[u], oof[v], a.n_boot, return_details=True)
                tests.append(dict(cohort=n, label=lb, a=u, b=v, block=blk, note=note,
                                  delta=d, p=p, ci_lo=ci[0], ci_hi=ci[1], n=int(len(y)), **detail))
                print(f"  {v:<26} vs {u:<22} delta {d:+.4f} "
                      f"[{ci[0]:+.4f},{ci[1]:+.4f}] p {p:.4f}"
                      f"{' *' if p < 0.05 else '  '} {note}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "args": vars(a), "scores": rows,
                                      "tests": tests, "predictions": predictions},
                                      indent=1))

    # --- combination is deliberately NOT done here --------------------------------
    print("\n" + "=" * 92)
    print("PER-COHORT RESULTS WRITTEN. Combination across trials is in")
    print("scripts/rct_clinical_meta.py, which pools delta/se_bootstrap by inverse variance.")
    print("It is a separate script because the obvious combination -- the study's weighted")
    print("Fisher-z Stouffer -- is WRONG for a paired difference of correlations and")
    print("understated one cell here by a factor of 4. See its module docstring.")
    print("=" * 92)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
