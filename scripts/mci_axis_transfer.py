#!/usr/bin/env python3
"""Fit the transfer axis inside MCI, not on the CN/Dementia extremes.

The mismatch this tests. Every ADNI-fitted axis in this project so far was fitted on CN
versus Dementia -- the two ENDS of the severity range -- and then applied to RCT
participants who are all in the MIDDLE: ACT and CogTE are MCI throughout, BREATHE is mixed
CN/MCI. A boundary fitted on the endpoints is an extrapolation where it is actually used.
`prepare_adni.py`'s ORDINAL note already made that argument; the build it implies was never
made, because `--label_col DX` drops MCI outright -- 686 of 1742 usable sessions.

`--dx_filter MCI` with a continuous cognitive target makes both the population and the
construct match the RCT side:

    population   655 MCI scans (299 subjects) instead of 994 CN+AD
    construct    ADNI_MEM / ADNI_EF level, the same composites the trials measure,
                 instead of a diagnosis label
    variance     MCI has the LARGEST within-group cognitive spread of the three stages
                 (ADNI_MEM sd 0.840 vs 0.742 CN and 0.825 Dementia), so restricting to it
                 does not restrict range as much as it might appear

Two arms are reported for every feature family, differing ONLY in what the axis was fitted
on, so the comparison is the population/construct change and nothing else:

    CN/AD      the published axis: binary DX on 994 CN+AD scans
    MCI-cog    ADNI_MEM or ADNI_EF regression on 655 MCI scans

Three things are measured after transfer. `level` is the axis score at T1 against the RCT's
baseline composite -- the target class the corrected meta found most promising (median
Stouffer Z 1.08, versus 0.18 for change). `change` is the T2-T1 axis difference against the
RCT's composite change, the actual intervention question. `drift` is the sd of the axis
difference over the entire ADNI fitted range, i.e. the reliability ratio, which decides
whether `change` could show anything at all.

An in-ADNI cross-validated score is printed first, and it has to be read before the
transfer numbers: within MCI the between-group variance that carries most of an all-stage
correlation is gone, so the achievable rho is much lower than the 0.38-0.39 the published
all-stage table reports for FC. If nothing predicts cognition within MCI, no transfer can
work and the RCT columns are uninterpretable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from benchmark_evaluation import PROTOCOL_VERSION, permutation_tail
from cohorts import center_rois, load_cohort  # noqa: E402
from datasets import load_xyz  # noqa: E402
from paths import (DERIVED_DIR, OFFICIAL_BRAINLM_13M, RUNS_DIR,  # noqa: E402
                   XYZ_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Axis:
    """Standardise -> PCA -> linear fit, with the transform frozen for transfer.

    Scaling before reduction, not after: PCA followed by StandardScaler rescales every
    component to unit variance, which flattens the eigenvalue spectrum and weights trailing
    noise like leading signal. That ordering cost 0.27 AUC when it was a live bug here
    (the historical correction record).
    """

    def __init__(self, n_pca=200, binary=False, alpha=1.0):
        self.n_pca, self.binary, self.alpha = n_pca, binary, alpha

    def fit(self, F, y, fit_mask=None):
        """`fit_mask` splits WHERE the transform is estimated from WHERE the slope is.

        The unsupervised half -- scaler and PCA basis -- is estimated on every row given,
        while the linear slope uses only `fit_mask`. That separation exists because the
        two halves have different bottlenecks here: an 89676-dim FC feature is badly
        conditioned at n=655, which more scans fix, but pooling diagnostic stages into the
        SLOPE imports the between-stage gradient, and half of ADNI_MEM's variance is
        between-stage (eta^2 0.497). Fitting the basis on all 1658 and the slope on the
        655 MCI rows takes the conditioning without the gradient.
        """
        self.sc = StandardScaler().fit(F)
        Z = self.sc.transform(F)
        self.pca = (PCA(n_components=min(self.n_pca, len(F) - 1), random_state=0).fit(Z)
                    if F.shape[1] > 2000 else None)
        if self.pca is not None:
            Z = self.pca.transform(Z)
        if fit_mask is not None:
            Z, y = Z[fit_mask], y[fit_mask]
        self.m = (LogisticRegression(C=0.01, max_iter=5000).fit(Z, y.astype(int))
                  if self.binary else Ridge(alpha=self.alpha).fit(Z, y))
        return self

    def score(self, F):
        Z = self.sc.transform(F)
        if self.pca is not None:
            Z = self.pca.transform(Z)
        return (self.m.decision_function(Z) if self.binary else self.m.predict(Z))


def cv_axis(F, y, groups, binary, n_pca=200, k=5, fit_mask=None, score_mask=None,
            raw_scans=None, stages=None):
    """Subject-grouped out-of-fold score, so the in-ADNI number is not inflated by a
    subject appearing in both folds -- ADNI contributes 2.2 sessions per subject here."""
    oof = np.full(len(y), np.nan)
    target = np.asarray(y, float).copy()
    for tr, te in GroupKFold(n_splits=k).split(F, y, groups):
        fm = None if fit_mask is None else fit_mask[tr]
        if fm is not None and fm.sum() < 20:
            continue
        train_y, test_y = y[tr].copy(), y[te].copy()
        if stages is not None:
            for st in np.unique(stages[te]):
                if not np.any(stages[tr] == st):
                    raise ValueError(f"Stage {st} absent from training fold")
            for st in np.unique(stages[tr]):
                mu = y[tr][stages[tr] == st].mean()
                train_y[stages[tr] == st] -= mu
                test_y[stages[te] == st] -= mu
        target[te] = test_y
        ftr, fte = F[tr], F[te]
        if raw_scans is not None:
            from thirdorder import transfer_features
            _, (ftr, fte) = transfer_features(raw_scans[tr],
                                             [raw_scans[tr], raw_scans[te]])
        a = Axis(n_pca, binary).fit(ftr, train_y, fm)
        oof[te] = a.score(fte)
    # The in-source score is reported on `score_mask` when the arm's question is about a
    # subpopulation: an all-stage arm scored on all stages is inflated by the between-stage
    # gradient, which is exactly the component that cannot transfer to a single-stage RCT.
    sm = np.ones(len(y), bool) if score_mask is None else score_mask
    sm = sm & np.isfinite(oof)
    if binary:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(target[sm].astype(int), oof[sm])), "AUC"
    return float(stats.spearmanr(target[sm], oof[sm]).statistic), "rho"


def adni_features(kind, x, model, xyz, readout, layer, rank, basis=None):
    from rct_probe import features
    return features(kind, x, None, model, xyz, readout, layer, x.shape[2], rank, basis, None)


def apply_gain_match(co, name=""):
    """Divide each session by ONE scalar -- its median per-ROI sd -- in place.

    Not a new normalisation: exactly what prepare_adni.py --amplitude preserve does. It
    removes the arbitrary BOLD gain while KEEPING the between-ROI amplitude ratios BrainLM
    was pretrained on, which per-ROI z-scoring would destroy (and section 4b showed that
    amplitude carries real signal).

    REQUIRED FOR ANY PSOPT-VS-OTHERS COMPARISON, and `--gsr` alone is not enough. Both psopt
    builds are raw parcel intensity: median per-ROI sd 6.23 (noGSR) / 5.25 (GSR) against
    0.679-0.733 for act/cogte/breathe and 0.521 for the ADNI _off build. Measured on
    fingerprinting, where the two corrections were separated:

        roi_var L4 top1     as delivered   gain only   gain + GSR
        psopt                   0.205        0.500        0.875
        act / cogte / breathe   0.874/0.885/0.949  ->  0.837/0.966/0.935

    So gain alone recovers less than half the gap and only the pair closes it, which is why
    both flags exist and why neither is a substitute for the other.
    """
    v = np.asarray(co.valid)
    before = []
    for i in range(co.bold.shape[0]):
        for s_ in range(co.bold.shape[3]):
            L = int(v[i, s_])
            sd = co.bold[i, :, :L, s_].std(dim=1, unbiased=False)
            g = float(sd.median().clamp_min(1e-6))
            before.append(g)
            co.bold[i, :, :, s_] = co.bold[i, :, :, s_] / g
    print(f"  --gain_match {name}: median per-ROI sd was {np.median(before):.3f} -> 1.000",
          flush=True)
    return co


def apply_gsr(co, name=""):
    """Regress each session's global mean timeseries out of every ROI, in place.

    A CONTROL for a release difference, not a preprocessing preference. The four RCT A424
    releases do not carry the same denoising -- paths.py records global-signal/per-ROI-sd as
    0.152 (act) / 0.248 (cogte) / 0.152 (breathe) with the note "the A424 release had global
    signal suppressed", while psopt's A424 measures 0.711 and its coverage.json says
    `noGSR: true`. ADNI's own _off build measures 0.351.

    This exists because that difference is a live alternative explanation for psopt's FC
    result. With the frozen encoder, `FC -> ef_level_t2` on the `all-EF resid` arm is 0.575
    (psopt) / 0.296 (cogte) / 0.171 (breathe) / 0.049 (act), which orders monotonically by
    retained global signal. It is NOT a source-target matching effect: psopt has the LARGEST
    gap from ADNI's 0.351 and the BEST transfer, which is the opposite of what matching
    predicts. And it is not the per-subject scalar either -- within psopt, global-signal
    strength correlates -0.113 (p=0.47) with ef_T2 and mean FD -0.252 (p=0.10), neither of
    which can manufacture a +0.59. So the remaining candidate is that retaining the global
    signal changes the FC PATTERN in a way that transfers, and only removing it decides.
    """
    v = np.asarray(co.valid)
    before, after = [], []
    for i in range(co.bold.shape[0]):
        for s in range(co.bold.shape[3]):
            L = int(v[i, s])
            x = co.bold[i, :, :L, s]
            ok = torch.isfinite(x).all(1) & (x.std(dim=1) > 1e-6)
            if int(ok.sum()) < 50:
                continue
            gs = x[ok].mean(dim=0)
            gs = gs - gs.mean()
            den = (gs * gs).sum().clamp_min(1e-12)
            xc = x - x.mean(dim=1, keepdim=True)
            before.append(float(gs.std() / x[ok].std(dim=1).median()))
            co.bold[i, :, :L, s] = xc - torch.outer((xc @ gs) / den, gs)
            y = co.bold[i, :, :L, s][ok]
            after.append(float(y.mean(dim=0).std() / y.std(dim=1).median()))
    print(f"  --gsr {name}: global-signal/per-ROI-sd "
          f"{np.median(before):.3f} -> {np.median(after):.2e}", flush=True)
    return co


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mci_dir", default=str(DERIVED_DIR / "adni424_t200_mci"))
    ap.add_argument("--cnad_dir", default=str(DERIVED_DIR / "adni424_t200"))
    ap.add_argument("--cnad_tag", default="raw_DX_all_t200_off")
    ap.add_argument("--tlen", type=int, default=200)
    ap.add_argument("--cohorts", nargs="+", default=["act", "cogte", "breathe"])
    ap.add_argument("--families", nargs="*", default=None,
                    help="restrict SPECS to these labels (default: all five)")
    ap.add_argument("--gsr", action="store_true",
                    help="Regress out each session's global signal before anything else. "
                         "See apply_gsr's docstring: the four RCT A424 releases do not "
                         "carry the same denoising (GS/per-ROI-sd 0.152/0.248/0.152 for "
                         "act/cogte/breathe against 0.711 for psopt), and that is the live "
                         "alternative explanation for psopt's FC transfer result. Default "
                         "off, so every published invocation is unchanged.")
    ap.add_argument("--gain_match", action="store_true",
                    help="Divide each session by its median per-ROI sd. See apply_gain_match: psopt is raw parcel intensity (per-ROI sd 6.23 against 0.68-0.73), and on fingerprinting gain alone moved psopt from 0.205 to 0.500 while gain+GSR reached 0.875, so BOTH corrections are needed for a psopt-vs-others comparison. Default off, so published invocations are unchanged.")
    ap.add_argument("--n_pca", type=int, default=200)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M),
                    help="BrainLM checkpoint for the encoder families. Point this at a "
                         "fine-tuned encoder (finetune_adni_probe.py --fit_all "
                         "--save_encoder) to ask whether post-training on the MATCHED ADNI "
                         "task improves this transfer; the default is the frozen official "
                         "release, which is what the published rows used.")
    ap.add_argument("--release", default="amp", choices=("amp", "zscore", "schaefer", "amp_nogsr", "zscore_nogsr", "schaefer_nogsr"),
                    help="Which stored file to read. The *_nogsr variants exist only for psopt, "
                         "which ships the same recordings under two denoisings "
                         "(pSOPT_visits12_GSR / pSOPT_visits12_noGSR, delivered "
                         "2026-09-17). Default `amp` is the GSR build, matching what "
                         "act/cogte/breathe's A424 releases are.")
    ap.add_argument("--out", default=str(RUNS_DIR / "benchmark_v1/mci_axis_transfer.json"))
    a = ap.parse_args()
    C.configure_backends()
    C.NUM_TIMEPOINTS_PER_VOXEL = a.tlen
    from thirdorder import shared_coskew_basis

    #: (label, kind, readout, layer). A424 only: the MCI build is A424, so Brain-JEPA's
    #: Schaefer parcellation has no matching ADNI side here.
    SPECS = [("FC", "fc", None, None), ("tucker-FC", "tucker", None, None),
             ("amp", "amp", None, None),
             ("BrainLM 13M roi_norm L4", "brainlm", "roi_norm", 4),
             ("BrainLM 13M emb_fc L3", "brainlm", "emb_fc", 3)]
    # `--families` exists for the alpha-scan (scripts/alpha_scale.py), which reruns this
    # script once per interpolated checkpoint. FC, tucker-FC and amp do not read the
    # encoder at all, so recomputing them at every alpha would burn the coskew basis
    # repeatedly to reproduce identical numbers.
    if a.families:
        keep = set(a.families)
        unknown = keep - {s[0] for s in SPECS}
        if unknown:
            raise SystemExit(f"unknown families: {sorted(unknown)}")
        SPECS = [s for s in SPECS if s[0] in keep]

    def load(d, tag):
        x = torch.load(f"{d}/ADNI_x_{tag}.pt", map_location="cpu", weights_only=False).float()
        y = torch.load(f"{d}/ADNI_y_{tag}.pt", map_location="cpu", weights_only=False).numpy()
        import pandas as pd
        man = pd.read_csv(f"{d}/ADNI_manifest_{tag}.csv")
        stage = (man["dx_stage"].astype(str).values if "dx_stage" in man.columns
                 else np.array(["?"] * len(man)))
        return x, y.astype(float), man["PTID"].values, stage

    #: (dir, tag) -> list of arm configs sharing that tensor, so its features are
    #: extracted once. `y_mode`: raw | resid (subtract each stage's mean) | binary.
    #: `fit_stage`: restrict the SLOPE to one stage while the basis still sees every row.
    SOURCES = [(a.cnad_dir, a.cnad_tag,
                [("CN/AD", "binary", None, None)])]
    for t in ("ADNI_MEM", "ADNI_EF"):
        short = t.split("_")[1]
        SOURCES.append((a.mci_dir, f"raw_{t}_all_t200_off_dxMCI",
                        [(f"MCI-{short}", "raw", None, None)]))
        SOURCES.append((a.cnad_dir, f"raw_{t}_all_t200_off", [
            # The diagnosis gradient, left in. Its in-source number is the inflated one and
            # the gap to `resid` measures how much of it is stage rather than cognition.
            (f"all-{short} raw", "raw", None, None),
            # Same 1658 rows, stage means removed: 2.5x the MCI-only sample for a 6% loss
            # of target spread (sd 0.793 vs 0.840 on MEM).
            (f"all-{short} resid", "resid", None, None),
            # Basis on all 1658, slope on the 655 MCI rows only.
            (f"all-{short} basisAll/fitMCI", "raw", "MCI", "MCI"),
        ]))

    # RCT scans once, centred, at the pretrained length
    from rct_probe import upsample_short
    rct = {}
    for n in a.cohorts:
        co = center_rois(load_cohort(n, release=a.release, require_labels=False))
        if a.gsr:
            co = apply_gsr(co, n)
        if a.gain_match:
            # AFTER gsr: the regression is scale-equivariant, so the order does not change
            # the result, but doing the gain last makes the printed median exactly 1.000.
            co = apply_gain_match(co, n)
        if int(co.valid.min()) < a.tlen:
            co, _ = upsample_short(co, a.tlen, C.TIMEPOINT_PATCHING_SIZE, fill="pad")
        rct[n] = co
    xyz = load_xyz(XYZ_PATH)

    rows = []
    for label, kind, readout, layer in SPECS:
        model = None
        if kind == "brainlm":
            gpu_acquire()
            from model_loader import load_brainlm_forpretraining
            model = load_brainlm_forpretraining(Path(a.pretrained)).to(DEVICE).eval()
        # RCT features once per family, both timepoints
        rf = {}
        if kind != "tucker":
            for n, co in rct.items():
                rf[n] = {tp: adni_features(kind, co.bold[..., tp][:, :, :a.tlen],
                                          model, xyz, readout, layer, 64)
                         for tp in (0, 1)}
        if kind == "brainlm": gpu_release()
        for d, tag, configs in SOURCES:
            xa, ya, ptid, stage = load(d, tag)
            bas = (shared_coskew_basis(xa.numpy(), None, 64) if kind == "tucker" else None)
            if kind == "brainlm": gpu_acquire()
            Fa = adni_features(kind, xa, model, xyz, readout, layer, 64, bas)
            if kind == "brainlm": gpu_release()
            if kind == "tucker":
                # Source-fitted basis is also used for BOTH target visits.
                rf = {n: {tp: adni_features(kind, co.bold[..., tp][:, :, :a.tlen],
                                             model, xyz, readout, layer, 64, bas)
                          for tp in (0, 1)} for n, co in rct.items()}
            for arm, y_mode, fit_stage, score_stage in configs:
              binary = y_mode == "binary"
              y_arm = ya.astype(float).copy()
              if y_mode == "resid":
                  # Subtract each stage's own mean. Removes the between-stage component --
                  # 50% of ADNI_MEM's variance -- and keeps the within-stage part, which is
                  # the only part with variance inside a single-stage RCT.
                  for st in np.unique(stage):
                      m = stage == st
                      y_arm[m] -= y_arm[m].mean()
              fm = None if fit_stage is None else (stage == fit_stage)
              sm = None if score_stage is None else (stage == score_stage)
              cv, metric = cv_axis(Fa, ya if y_mode == "resid" else y_arm,
                                   ptid, binary, a.n_pca, fit_mask=fm, score_mask=sm,
                                   raw_scans=xa.numpy() if kind == "tucker" else None,
                                   stages=stage if y_mode == "resid" else None)
              ax = Axis(a.n_pca, binary).fit(Fa, y_arm, fm)
              s = ax.score(Fa)
              spread = float(s.std())
              for n, co in rct.items():
                s1, s2 = ax.score(rf[n][0]), ax.score(rf[n][1])
                drift = float(np.std(s2 - s1)) / max(spread, 1e-12)
                out = {"family": label, "arm": arm, "adni_n": int(len(ya)),
                       "adni_cv": cv, "adni_metric": metric, "cohort": n,
                       "y_mode": y_mode, "fit_stage": fit_stage,
                       "expected_direction": -1 if binary else 1,
                       "drift_over_adni_sd": drift}
                # Three targets per label, and `level_t2` is the point of the rerun. The
                # T1 cross-sectional correlation does not replicate at T2 -- 12 of the
                # 12 significant T1 cells in the main table, only 2 survive at T2, and 12
                # of 60 pairs flip sign by more than 0.2. So a sign pattern that appears
                # only at T1 is a session effect, and the pattern this script found has to
                # be shown at both sessions or withdrawn.
                #
                # `labelled_mask`, not isfinite: the first version of this script used
                # isfinite alone and got BREATHE n=107 against the published 81, so its
                # BREATHE column -- which carried most of the combined Z -- was not
                # comparable to anything.
                for lb in ("mem", "ef"):
                    base = co.labelled_mask(lb)
                    for tname, yv, pred_all in (
                            ("level", co.baseline_levels(lb), s1),
                            ("level_t2", co.followup_levels(lb), s2),
                            ("change", co.change_scores(lb), s2 - s1)):
                        yv = np.asarray(yv, float)
                        m = base & np.isfinite(yv)
                        if int(m.sum()) < 20:
                            continue
                        pred = pred_all[m]
                        r = float(stats.spearmanr(yv[m], pred).statistic)
                        rr = np.random.default_rng(0)
                        nul = np.array([stats.spearmanr(rr.permutation(yv[m]), pred).statistic
                                        for _ in range(a.n_perm)])
                        out[f"{lb}_{tname}_rho"] = r
                        out[f"{lb}_{tname}_p"] = permutation_tail(r, nul, two_sided=True)
                        out[f"{lb}_{tname}_n"] = int(m.sum())
                rows.append(out)
                print(f"  {label:<24}{arm:<26}{n:<8} ADNI {metric} {cv:+.4f} (n={len(ya)})"
                      f"  drift {drift:.3f}"
                      f"  memT1 {out.get('mem_level_rho', float('nan')):+.3f}"
                      f"  memT2 {out.get('mem_level_t2_rho', float('nan')):+.3f}"
                      f"  memChg {out.get('mem_change_rho', float('nan')):+.3f}"
                      f"  efT1 {out.get('ef_level_rho', float('nan')):+.3f}"
                      f"  efT2 {out.get('ef_level_t2_rho', float('nan')):+.3f}",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"protocol_version": PROTOCOL_VERSION,
                                     "args": vars(a), "rows": rows}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
