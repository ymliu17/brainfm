#!/usr/bin/env python3
"""Sweep layers x readouts for the CN/AD axis. Diagnoses WHY the encoder loses to FC.

THE OBSERVATION THIS IS BUILT ON
--------------------------------
On ADNI CN-vs-AD the frozen encoder reaches 0.630 while two trivial features beat it:
FC edges 0.723 and the 424-dim per-ROI amplitude profile 0.682. Both winners are
ROI-RESOLVED -- their whole content is which region differs from which. Every readout tried
so far (`cls`, `mean`, `mean_last2`, ...) averages over the entire token sequence, and the
token layout is ROI-major:

    token index 1 + roi * num_patches + patch      (BrainLMEmbeddings, modeling_brainlm.py)

so `mean` sums across all 424 ROIs and destroys exactly the axis the winners exploit. The
information is not necessarily absent from the encoder; the readout may be discarding it.
This script tests that, and sweeps the layer at the same time -- an MAE's last layers
specialise for reconstruction, so mid-depth features often probe better, and no layer sweep
had been run here.

READOUTS
--------
    cls        the CLS token                                          [hidden]
    mean       mean over all patch tokens (the historical default)    [hidden]
    roi_norm   L2 norm of each ROI's mean token -> a spatial map      [424]
    roi_var    each ROI's variance across its time patches,
               averaged over hidden dims -> an "embedding amplitude
               profile", the direct analogue of the winning feature   [424]
    emb_fc     cosine similarity between ROIs' mean tokens, upper
               triangle -> "FC in embedding space", the direct
               analogue of the winning FC feature                     [424*423/2]

`roi_var` and `emb_fc` are the two that ask the real question: if the encoder's own
ROI-resolved summaries beat its pooled ones, the encoder was never the problem -- the
readout was.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from datasets import load_xyz  # noqa: E402
from model_loader import load_brainlm_forpretraining  # noqa: E402
from paths import BRAINLM_111M, DERIVED_DIR, OFFICIAL_BRAINLM_13M, RUNS_DIR, XYZ_PATH  # noqa: E402
from vitmae_input import MAX_VAL_TO_SCALE, PAD_VALUE, load_vitmae, prepare_batch  # noqa: E402
from brainlm_input import roi_order_by_y  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHEAP = ("cls", "mean", "roi_norm", "roi_var")


def _encoder(model):
    return model.vit if hasattr(model, "vit") else model.base_model.model.vit


@torch.no_grad()
def extract(model, x, xyz, layers, readouts, batch=8):
    """{(layer, readout): [N, d]} in one pass over the data.

    Hidden states for every layer cannot be stored (13 x 2969 x 512 x 498 floats is ~39 GB),
    so each readout is reduced to its compact form inside the batch loop.
    """
    n_patch = x.shape[2] // C.TIMEPOINT_PATCHING_SIZE
    n_roi = x.shape[1]
    acc: dict = {}
    for s in range(0, len(x), batch):
        xb = x[s:s + batch].to(DEVICE)
        out = _encoder(model)(signal_vectors=xb,
                              xyz_vectors=xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1).to(DEVICE),
                              noise=None, return_dict=True, output_hidden_states=True)
        for li in layers:
            h = out.hidden_states[li]                       # [B, 1+roi*patch, hid]
            cls, tok = h[:, 0, :], h[:, 1:, :]
            g = tok.reshape(tok.shape[0], n_roi, n_patch, tok.shape[-1])   # ROI-major
            roi_mean = g.mean(dim=2)                        # [B, roi, hid]
            for r in readouts:
                if r == "cls":
                    v = cls
                elif r == "mean":
                    v = tok.mean(dim=1)
                elif r == "roi_norm":
                    v = roi_mean.norm(dim=-1)               # [B, roi]
                elif r == "roi_var":
                    v = g.var(dim=2, unbiased=False).mean(dim=-1)          # [B, roi]
                elif r == "emb_fc":
                    z = roi_mean - roi_mean.mean(dim=-1, keepdim=True)
                    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    cm = z @ z.transpose(1, 2)              # [B, roi, roi]
                    iu = torch.triu_indices(n_roi, n_roi, offset=1, device=cm.device)
                    v = cm[:, iu[0], iu[1]]
                else:
                    raise ValueError(r)
                acc.setdefault((li, r), []).append(v.float().cpu())
        print(f"  {min(s + batch, len(x))}/{len(x)}", end="\r", flush=True)
    print()
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


@torch.no_grad()
def brainlm_readout_batched(model, x3, xyz, layer, readout, batch=8):
    """One (layer, readout) for the 13M path, as a [N, d] array. Wraps `extract`."""
    n_layers = len(_encoder(model).encoder.layer)
    li = n_layers if layer < 0 else layer
    return extract(model, x3, xyz, [li], [readout], batch)[(li, readout)]


def grid_geometry(image_size, patch, n_roi, n_time):
    """Which cells of the ViTMAE patch grid contain real signal rather than padding.

    `prepare_batch` centres a [n_roi, n_time] recording in a square image and pads the rest
    with the constant PAD_VALUE. At n_time=140 in a 432-wide image that is 146 columns of
    padding on each side, so only 9 of the 27 time-blocks touch real data and 486 of the
    729 tokens are pure padding. A `mean` readout over all tokens is therefore two thirds
    padding constant -- which is not a tuning detail, it is most of the signal budget.
    """
    h = w = image_size[0] if isinstance(image_size, (list, tuple)) else image_size
    gh, gw = h // patch, w // patch
    pad_top, pad_left = (h - n_roi) // 2, (w - n_time) // 2
    rows = [i for i in range(gh) if (i + 1) * patch > pad_top and i * patch < pad_top + n_roi]
    cols = [j for j in range(gw) if (j + 1) * patch > pad_left and j * patch < pad_left + n_time]
    return gh, gw, rows, cols


@torch.no_grad()
def extract_vitmae(model, cfg, x, order, layers, readouts, batch=8):
    """Same readout family for the ViTMAE releases, on their [ROI-block, time-block] grid.

    Two differences from the 13M that are properties of the architecture, not choices:
      * a token spans `patch` ROIs x `patch` timepoints, so the finest spatial resolution
        available is 27 ROI-blocks (111M), not 424 ROIs. `emb_fc` is 351 edges, not 89676.
      * most tokens are padding at short recording lengths, hence the `_real` variants,
        which pool only over the grid cells that contain signal.
    """
    n_patches = model.vit.embeddings.patch_embeddings.num_patches
    gh, gw, rows, cols = grid_geometry(cfg.image_size, cfg.patch_size,
                                       x.shape[1], x.shape[2])
    print(f"  grid {gh}x{gw}; ROI-blocks with signal {len(rows)}/{gh}, "
          f"time-blocks with signal {len(cols)}/{gw} "
          f"({100 * (1 - len(rows) * len(cols) / (gh * gw)):.0f}% of tokens are pure padding)",
          flush=True)
    ri = torch.as_tensor(rows)
    ci = torch.as_tensor(cols)
    acc: dict = {}
    for s0 in range(0, len(x), batch):
        pv = prepare_batch(x[s0:s0 + batch], order, cfg.image_size).to(DEVICE)
        noise = torch.zeros(pv.shape[0], n_patches, device=pv.device)
        out = model.vit(pixel_values=pv, noise=noise, output_hidden_states=True,
                        return_dict=True)
        for li in layers:
            h = out.hidden_states[li]
            cls, tok = h[:, 0, :], h[:, 1:, :]
            g = tok.reshape(tok.shape[0], gh, gw, tok.shape[-1])     # [B, roi_blk, t_blk, hid]
            gr = g[:, ri][:, :, ci]                                   # signal-bearing cells
            roi_mean = gr.mean(dim=2)                                  # [B, n_rows, hid]
            for r in readouts:
                if r == "cls":
                    v = cls
                elif r == "mean":
                    v = tok.mean(dim=1)
                elif r == "mean_real":
                    v = gr.reshape(gr.shape[0], -1, gr.shape[-1]).mean(dim=1)
                elif r == "roi_norm":
                    v = roi_mean.norm(dim=-1)
                elif r == "roi_var":
                    v = gr.var(dim=2, unbiased=False).mean(dim=-1)
                elif r == "emb_fc":
                    z = roi_mean - roi_mean.mean(dim=-1, keepdim=True)
                    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    cm = z @ z.transpose(1, 2)
                    n = cm.shape[1]
                    iu = torch.triu_indices(n, n, offset=1, device=cm.device)
                    v = cm[:, iu[0], iu[1]]
                else:
                    raise ValueError(r)
                acc.setdefault((li, r), []).append(v.float().cpu())
        print(f"  {min(s0 + batch, len(x))}/{len(x)}", end="\r", flush=True)
    print()
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def probe(F, y, seeds=(0, 1, 2), n_pca=200, grid=(0.001, 0.01, 0.1, 1.0, 10.0),
          groups=None, covariates=None, site_out=None, n_site_folds=5):
    """Probe score, hyperparameter and PCA fitted inside each training fold.

    Binary target -> logistic probe scored by AUC. Ordinal target (the graded DX severity
    in [0, 1], which keeps MCI) -> ridge scored by Spearman rho, because AUC cannot express
    a three-level ordering. Folds are stratified on the discrete level either way, so the
    rarest stage is represented in every fold.

    `groups` MUST be supplied for an all-sessions build. A subject contributes several
    sessions there, and two scans of one person resemble each other far more than anyone
    else's, so an ungrouped split leaves a subject's other sessions in the training set and
    the held-out score is inflated rather than honest. With groups the split is by subject;
    stratification is then approximate, achieved by dealing whole subjects into folds within
    each stratum.
    """
    levels = np.unique(y)
    ordinal = len(levels) > 2
    if not ordinal:
        strat = y.astype(int)
    elif len(levels) <= 10:
        # a genuinely ordinal target (CN / MCI / Dementia) -- stratify on the level itself
        strat = np.searchsorted(levels, y)
    else:
        # a continuous target. Stratifying on unique values would make one class per
        # subject: StratifiedKFold then either errors or, when duplicate values happen to
        # give every value >= n_splits members, silently produces near-arbitrary folds.
        # Quantile bins keep the target's distribution balanced across folds instead.
        strat = np.clip(np.searchsorted(np.quantile(y, [0.2, 0.4, 0.6, 0.8]), y), 0, 4)
    def splits(seed):
        if site_out is not None:
            # Leave-one-SITE-out. Stratification balances folds; it does not stop a model
            # from reading site instead of biology, and site is strongly associated with the
            # target here (chi2 p = 1.3e-18 across 48 sites in the all-sessions build) while
            # the amplitude profile alone predicts acquisition protocol at AUC 0.97. Held-out
            # sites are the only test of whether the score survives a scanner the model has
            # never seen. Sites are grouped into `n_site_folds` blocks of similar size,
            # because 48 sites with a median of 9-16 scans give held-out sets too small to
            # score one at a time.
            st = np.asarray(site_out)
            order = pd.Series(st).value_counts().index.tolist()   # largest first
            block = {}
            load = np.zeros(n_site_folds)
            for sname in order:                                   # greedy balance by size
                k = int(np.argmin(load))
                block[sname] = k
                load[k] += (st == sname).sum()
            b = np.array([block[v] for v in st])
            for k in range(n_site_folds):
                tr, te = np.where(b != k)[0], np.where(b == k)[0]
                if len(tr) > 20 and len(te) > 5 and (ordinal or len(set(y[te])) > 1):
                    yield tr, te
            return
        if groups is None:
            yield from StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=seed).split(F, strat)
            return
        g = np.asarray(groups)
        uniq = np.unique(g)
        # Stratify on the target AND on covariates that are associated with it, because a
        # fold imbalanced on those is imbalanced on the thing being predicted. Measured on
        # this data: site x label chi2 p = 3.8e-04 (first session) and 1.3e-18 (all
        # sessions) across 48 sites with a median of 9-16 scans each; sex r = +0.10; AD
        # subjects are 3 years older (p = 5e-05). `adni_folds.assign_folds` already
        # stratified on age/sex/education and this split did not, which was an
        # inconsistency between the two evaluation paths in this repo.
        sub_strat = {}
        for u in uniq:
            rows = np.where(g == u)[0]
            key = [int(np.bincount(strat[rows]).argmax())]
            if covariates is not None:
                key += [int(covariates[rows[0], j]) for j in range(covariates.shape[1])]
            sub_strat[u] = tuple(key)
        # Cells finer than n_splits cannot be spread across folds; collapse the rare ones to
        # the target-only key so stratification degrades gracefully instead of failing.
        from collections import Counter
        cnt = Counter(sub_strat.values())
        sub_strat = {u: (k if cnt[k] >= 5 else (k[0],)) for u, k in sub_strat.items()}
        rng = np.random.default_rng(seed)
        fold_of = {}
        for st in sorted(set(sub_strat.values())):
            for i, u in enumerate(rng.permutation([u for u in uniq if sub_strat[u] == st])):
                fold_of[u] = i % 5
        f = np.array([fold_of[u] for u in g])
        for k in range(5):
            tr, te = np.where(f != k)[0], np.where(f == k)[0]
            if len(tr) and len(te):
                yield tr, te

    out = []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        for tr, te in splits(seed):
            A, B = F[tr], F[te]
            # Scale BEFORE reducing. PCA followed by StandardScaler rescales every
            # component to unit variance, which flattens the eigenvalue spectrum and
            # gives the trailing noise directions the same weight as the leading signal
            # ones; on 94 components from 95 samples that cost 0.27 AUC (measured).
            sc = StandardScaler().fit(A)
            A, B = sc.transform(A), sc.transform(B)
            if F.shape[1] > 2000:
                p = PCA(n_components=min(n_pca, len(tr) - 1), random_state=0).fit(A)
                A, B = p.transform(A), p.transform(B)
            # inner split for C
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(A))
            cut = max(2, int(0.2 * len(A)))
            iv, it = perm[:cut], perm[cut:]
            best, bc = -np.inf, grid[0]
            for c in grid:
                if len(set(y[tr][it])) < 2 or len(set(y[tr][iv])) < 2:
                    continue
                if ordinal:
                    m = Ridge(alpha=1.0 / c).fit(A[it], y[tr][it])
                    a = stats.spearmanr(y[tr][iv], m.predict(A[iv])).statistic
                    a = a if np.isfinite(a) else -np.inf
                else:
                    m = LogisticRegression(C=c, max_iter=5000).fit(A[it], y[tr][it])
                    a = roc_auc_score(y[tr][iv], m.predict_proba(A[iv])[:, 1])
                if a > best:
                    best, bc = a, c
            if ordinal:
                oof[te] = Ridge(alpha=1.0 / bc).fit(A, y[tr]).predict(B)
            else:
                oof[te] = LogisticRegression(C=bc, max_iter=5000).fit(A, y[tr]).predict_proba(B)[:, 1]
        v = np.isfinite(oof)
        out.append(float(stats.spearmanr(y[v], oof[v]).statistic) if ordinal
                   else roc_auc_score(y[v], oof[v]))
    return float(np.mean(out)), float(np.std(out))


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description="Layer x readout sweep for the CN/AD axis")
    ap.add_argument("--kind", default="brainlm",
                    choices=("brainlm", "vitmae", "brainjepa"))
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M),
                    help="checkpoint file for --kind brainlm, model DIRECTORY for vitmae")
    ap.add_argument("--jepa_ckpt",
                    default="checkpoints/pretrained/brainjepa/jepa-ep300.pth.tar")
    ap.add_argument("--jepa_gradient",
                    default="checkpoints/pretrained/brainjepa/gradient_mapping_450.csv")
    ap.add_argument("--tag", default="raw_DX_first_t140_off")
    ap.add_argument("--in_dir", default=str(DERIVED_DIR / "adni"))
    ap.add_argument("--readouts", nargs="+", default=list(CHEAP))
    ap.add_argument("--layers", nargs="+", default=["all"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--leave_site_out", action="store_true",
                    help="evaluate on held-out SITES instead of held-out subjects. Sites are "
                         "grouped into 5 size-balanced blocks. This is a generalisation test, "
                         "not a variance reduction: it asks whether the score survives a "
                         "scanner the model never saw.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    C.configure_backends()

    d = Path(args.in_dir)
    x = torch.load(d / f"ADNI_x_{args.tag}.pt", map_location="cpu", weights_only=False).float()
    y = torch.load(d / f"ADNI_y_{args.tag}.pt", map_location="cpu", weights_only=False).numpy()
    xyz = load_xyz(XYZ_PATH)
    gpu_acquire()
    if args.kind == "brainjepa":
        # Fixed 450 ROIs x 160 frames; both are baked into the released position embedding
        # (4500 rows), so the tag must be a 450-ROI t160 build.
        from brainjepa import load_brainjepa, N_ROI, SEQ_LEN
        if x.shape[1] != N_ROI or x.shape[2] < SEQ_LEN:
            raise SystemExit(
                f"--kind brainjepa needs [N, {N_ROI}, >={SEQ_LEN}]; {args.tag} is "
                f"{tuple(x.shape)}. Build it with prepare_adni.py --n_roi 450 --tlen 160.")
        model = load_brainjepa(Path(args.jepa_ckpt), Path(args.jepa_gradient)).to(
            DEVICE).eval()
        n_layers = len(model.blocks)
    elif args.kind == "vitmae":
        model, cfg = load_vitmae(Path(args.pretrained), device=DEVICE)
        n_layers = cfg.num_hidden_layers
    else:
        model = load_brainlm_forpretraining(Path(args.pretrained)).to(DEVICE).eval()
        n_layers = len(_encoder(model).encoder.layer)
    layers = list(range(n_layers + 1)) if args.layers == ["all"] else [int(v) for v in args.layers]
    bad = [v for v in layers if v > n_layers]
    if bad:
        raise SystemExit(f"--layers {bad} exceed this encoder's depth "
                         f"({n_layers} layers, so valid indices are 0..{n_layers})")
    print(f"{args.tag}: X={tuple(x.shape)}  AD {int(y.sum())} / CN {int((1 - y).sum())}")
    print(f"{args.kind} encoder has {n_layers} layers; hidden_states index 0 = embeddings, "
          f"{n_layers} = final. Sweeping {layers}")
    print(f"readouts: {args.readouts}", flush=True)

    # The manifest drives fold grouping and is needed for EVERY encoder kind. It used to
    # be assigned only inside the brainlm branch, so any other --kind raised
    # UnboundLocalError here; and the feature call below was unconditional, so even when
    # that was survived the non-brainlm features were silently overwritten by the brainlm
    # ones. Both are fixed by hoisting the manifest and dispatching features once, after.
    man_p = Path(args.in_dir) / f"ADNI_manifest_{args.tag}.csv"
    groups, covariates, site_out = None, None, None
    if man_p.exists():
        mm = pd.read_csv(man_p)
        if "PTID" in mm.columns and mm["PTID"].nunique() < len(mm):
            groups = mm["PTID"].values
            print(f"all-sessions build ({len(mm)} rows / {mm['PTID'].nunique()} subjects): "
                  f"folds grouped by subject", flush=True)
        cov_cols, cov = [], []
        # Site is coarsened to a tertile of its own size: 48 sites with a median of 9-16
        # scans cannot each appear in all 5 folds, so stratifying on raw site id would
        # collapse to no stratification at all.
        if "SITE" in mm.columns:
            sz = mm["SITE"].map(mm["SITE"].value_counts())
            cov.append(pd.qcut(sz, 3, labels=False, duplicates="drop").fillna(0).values)
            cov_cols.append("site_size_tertile")
        if "PTGENDER" in mm.columns:
            cov.append((mm["PTGENDER"] == "Male").astype(int).values)
            cov_cols.append("sex")
        if "AGE" in mm.columns:
            cov.append(pd.qcut(pd.to_numeric(mm["AGE"], errors="coerce"), 3, labels=False,
                               duplicates="drop").fillna(0).values)
            cov_cols.append("age_tertile")
        covariates = np.column_stack(cov).astype(int) if cov else None
        if covariates is not None:
            print(f"  folds also stratified on {cov_cols}", flush=True)
        if args.leave_site_out:
            if "SITE" not in mm.columns:
                raise SystemExit("--leave_site_out needs a SITE column in the manifest")
            site_out = mm["SITE"].values
            print(f"  LEAVE-ONE-SITE-OUT over {mm['SITE'].nunique()} sites in 5 blocks; "
                  f"subject grouping and covariate stratification do not apply", flush=True)
        else:
            site_out = None
    if args.kind == "brainjepa":
        from brainjepa import jepa_readout
        # One block depth only: the released encoder exposes no intermediate readout that
        # matches the BrainLM layer sweep, so the final representation is what is scored.
        feats = {(n_layers, r): jepa_readout(model, x, r, batch=4) for r in args.readouts}
    elif args.kind == "vitmae":
        feats = extract_vitmae(model, cfg, x, roi_order_by_y(XYZ_PATH), layers,
                               args.readouts, args.batch_size)
    else:
        feats = extract(model, x, xyz, layers, args.readouts, args.batch_size)
    del model
    from benchmark_gpu_phase import release_gpu_for_cpu_tail
    release_gpu_for_cpu_tail()
    rows = []
    for (li, r), F in sorted(feats.items()):
        m, s = probe(F, y, groups=groups, covariates=covariates, site_out=site_out)
        rows.append({"layer": li, "readout": r, "dim": int(F.shape[1]),
                     "auc": m, "auc_sd": s})
        metric = "rho" if len(np.unique(y)) > 2 else "AUC"
        print(f"  layer {li:2d}  {r:9s} dim {F.shape[1]:6d}   {metric} {m:+.4f} +/- {s:.4f}",
              flush=True)

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 72}\nbest per readout")
    for r in args.readouts:
        sub = df[df.readout == r].sort_values("auc", ascending=False)
        if len(sub):
            b = sub.iloc[0]
            print(f"  {r:9s} best layer {int(b.layer):2d}  AUC {b.auc:+.4f} +/- {b.auc_sd:.4f}")
    print("\nreference on the same subjects (scripts/positive_control.py):")
    print("  FC edges 0.7230 | amplitude profile 0.6819 | historical best readout 0.6296")
    print("=" * 72)

    out = Path(args.out) if args.out else (RUNS_DIR / "v2" / "readout_sweep" /
                                           f"{Path(args.pretrained).stem}_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "rows": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
