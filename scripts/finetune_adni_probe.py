#!/usr/bin/env python3
"""Supervised CN/AD fine-tuning that starts from the frozen probe, not from a random head.

WHY THE PREVIOUS VERSION LOST TO A LINEAR PROBE
-----------------------------------------------
`finetune_adni.py` reached val AUC 0.5921 on the 498-subject build while a frozen linear
probe on the same encoder reaches 0.6846 out-of-fold. Three specific reasons, all measured:

  1. It reads the CLS token of the LAST encoder layer. The readout sweep
     (runs/v2/readout_sweep/) puts that combination at 0.6256 -- the worst of the four
     readouts -- while `emb_fc` at layer 3 gives 0.6846. The architecture ignored what the
     sweep established.
  2. It unfreezes the last encoder block, i.e. it spends its trainable capacity on the
     layer measured to be worst for this task. Layer 3 is where the signal is.
  3. Its 0.5921 is not a held-out estimate. It is the best of 50 epochs on the single
     validation split used for early stopping, so it is optimistically biased -- with 117
     positives that split holds ~23, and the AUC on 23 positives is very noisy. The true
     gap to the frozen probe is therefore wider than 0.09, not narrower.

WHAT THIS SCRIPT DOES DIFFERENTLY
---------------------------------
  * reads a configurable (layer, readout), defaulting to the sweep's optimum
  * initialises the head from the LOGISTIC PROBE fitted on the frozen features of the
    training fold. Fine-tuning therefore starts at the frozen probe's solution, which makes
    "fine-tuning cannot beat the frozen probe" a falsifiable claim rather than an artifact
    of a random head: at epoch 0 the model IS the frozen probe.
  * adapts LoRA only by default, and can unfreeze a chosen block rather than always the last
  * reports NESTED cross-validated out-of-fold AUC, which is the same protocol the frozen
    probe is scored under, so the two numbers are comparable. The inner split is used for
    early stopping only and its AUC is never reported as a result.

The epoch-0 out-of-fold AUC is printed alongside the final one. If fine-tuning helps, the
final exceeds epoch 0; if it does not, that is a clean measurement of the encoder's
adaptability rather than a statement about head initialisation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from adni_finetune import BrainLMADNIClassifier  # noqa: E402
from adni_folds import assign_folds  # noqa: E402
from datasets import load_xyz  # noqa: E402
from paths import DERIVED_DIR, OFFICIAL_BRAINLM_13M, RUNS_DIR, XYZ_PATH  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbeHeadClassifier(BrainLMADNIClassifier):
    """LoRA backbone + a chosen (layer, readout) + a LINEAR head, probe-initialisable.

    The head is linear rather than the inherited 2-layer MLP so that it can be set exactly
    to a fitted logistic probe's coefficients. A nonlinear head could not be initialised
    that way, and then "fine-tuning starts at the frozen probe" would not be true.
    """

    def __init__(self, pretrained_backbone_path=None, lora_r=None,
                 unfreeze_block: int | None = None, readout="emb_fc", layer=3,
                 n_pca=200):
        super().__init__(pretrained_backbone_path=pretrained_backbone_path, lora_r=lora_r,
                         unfreeze_last_block=False)
        self.readout, self.layer, self.n_pca = readout, layer, n_pca
        self.pca_mean = None      # set by fit_head; PCA is needed for the 89676-dim emb_fc
        self.pca_comp = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.head = nn.Linear(n_pca, 1)
        self.classifier = nn.Identity()
        if unfreeze_block is not None:
            vit = self.brainlm.base_model.model.vit
            n = len(vit.encoder.layer)
            idx = unfreeze_block if unfreeze_block >= 0 else n + unfreeze_block
            if not 0 <= idx < n:
                raise ValueError(f"--unfreeze_block {unfreeze_block} outside 0..{n - 1}")
            for name, p in self.brainlm.named_parameters():
                if f"encoder.layer.{idx}." in name:
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        """Keep the ViT in eval mode even while training.

        Two reasons, and the first is load-bearing:

        1. In train mode `BrainLMEmbeddings` masks a fraction of patches (mask_ratio 0.2),
           which shortens the token sequence to a length that is no longer 424 * patches --
           the ROI-resolved readouts cannot be formed from it at all.
        2. More importantly, the frozen probe this run is compared against is computed with
           masking off. If fine-tuning ran with masking on, the epoch-0 model would NOT be
           the frozen probe and the comparison would confound "adaptation helped" with "the
           forward pass changed". Masking is a pretraining augmentation; it is not part of
           the discriminative readout being measured.

        Dropout is off as a consequence. That is an accepted trade: capacity is already
        constrained by LoRA r=8 plus weight decay, and an exactly-comparable epoch 0 is
        worth more here than the extra regularisation.
        """
        super().train(mode)
        self.brainlm.eval()
        return self

    def features(self, signal_vectors, xyz_vectors):
        out = self.brainlm.base_model.model.vit(
            signal_vectors=signal_vectors, xyz_vectors=xyz_vectors,
            output_hidden_states=True, return_dict=True, noise=None)
        h = out.hidden_states[self.layer]
        cls, tok = h[:, 0, :], h[:, 1:, :]
        n_roi = signal_vectors.shape[1]
        n_patch = tok.shape[1] // n_roi
        g = tok.reshape(tok.shape[0], n_roi, n_patch, tok.shape[-1])
        roi_mean = g.mean(dim=2)
        if self.readout == "cls":
            return cls
        if self.readout == "mean":
            return tok.mean(dim=1)
        if self.readout == "roi_var":
            return g.var(dim=2, unbiased=False).mean(dim=-1)
        if self.readout == "roi_norm":
            return roi_mean.norm(dim=-1)
        z = roi_mean - roi_mean.mean(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cm = z @ z.transpose(1, 2)
        iu = torch.triu_indices(n_roi, n_roi, offset=1, device=cm.device)
        return cm[:, iu[0], iu[1]]

    def project(self, f):
        """Standardisation + PCA, as fixed tensors so gradients flow through the encoder."""
        if self.pca_comp is None:
            return f
        # Order must match fit_probe_into_head: standardise the raw features first, then
        # project. (The reverse -- project then standardise -- flattens the PCA spectrum.)
        return ((f - self.scaler_mean) / self.scaler_scale - self.pca_mean) @ self.pca_comp.T

    def forward(self, signal_vectors, xyz_vectors):
        return self.head(self.project(self.features(signal_vectors, xyz_vectors))).squeeze(-1)


@torch.no_grad()
def frozen_features(model, x, xyz, batch=8):
    out = []
    for s in range(0, len(x), batch):
        xb = x[s:s + batch].to(DEVICE)
        out.append(model.features(xb, xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1)).float().cpu())
    return torch.cat(out).numpy()


def _strata(y):
    """Fold/split strata for a binary, ordinal, or continuous target."""
    lv = np.unique(y)
    if len(lv) == 2:
        return y.astype(int)
    if len(lv) <= 10:
        return np.searchsorted(lv, y)
    return np.clip(np.searchsorted(np.quantile(y, [0.2, 0.4, 0.6, 0.8]), y), 0, 4)


def _score(y, pred):
    """AUC for a binary target, Spearman rho otherwise -- one metric per target type."""
    if len(np.unique(y)) == 2:
        return float(roc_auc_score(y, pred))
    r = stats.spearmanr(y, pred).statistic
    return float(r) if np.isfinite(r) else float("nan")


def fit_probe_into_head(model, Ftr, ytr, grid=(0.001, 0.01, 0.1, 1.0, 10.0), seed=0):
    """Fit PCA + a linear probe on the training fold and write it into the model.

    After this the model's forward pass reproduces the probe exactly, so training starts
    from the frozen probe rather than from noise.

    Logistic for a two-class target, ridge for an ordinal or continuous one. The ordinal
    branch exists because the graded DX target (CN / MCI / Dementia as 0 / 0.5 / 1) keeps
    MCI and so covers the middle of the axis, where every RCT participant sits -- but
    LogisticRegression rejects it outright, which is what made the first attempt fail.
    """
    k = min(model.n_pca, len(Ftr) - 1, Ftr.shape[1])
    # Scale BEFORE reducing: PCA -> StandardScaler would rescale every component to unit
    # variance, flattening the spectrum so trailing noise directions weigh as much as the
    # leading signal ones.
    sc = StandardScaler().fit(Ftr)
    pca = PCA(n_components=k, random_state=0).fit(sc.transform(Ftr))
    A = pca.transform(sc.transform(Ftr))
    # C chosen on an inner split of the training fold only
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(A))
    cut = max(2, int(0.2 * len(A)))
    iv, it = perm[:cut], perm[cut:]
    binary = len(np.unique(ytr)) == 2
    best, bc = -np.inf, grid[0]
    for c in grid:
        if len(set(ytr[it])) < 2 or len(set(ytr[iv])) < 2:
            continue
        if binary:
            m = LogisticRegression(C=c, max_iter=5000).fit(A[it], ytr[it])
            a = _score(ytr[iv], m.predict_proba(A[iv])[:, 1])
        else:
            m = Ridge(alpha=1.0 / c).fit(A[it], ytr[it])
            a = _score(ytr[iv], m.predict(A[iv]))
        if np.isfinite(a) and a > best:
            best, bc = a, c
    lr = (LogisticRegression(C=bc, max_iter=5000) if binary
          else Ridge(alpha=1.0 / bc)).fit(A, ytr)
    dev = DEVICE
    model.pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=dev)
    model.pca_comp = torch.tensor(pca.components_, dtype=torch.float32, device=dev)
    model.scaler_mean = torch.tensor(sc.mean_, dtype=torch.float32, device=dev)
    model.scaler_scale = torch.tensor(sc.scale_, dtype=torch.float32, device=dev)
    model.head = nn.Linear(k, 1).to(dev)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(np.atleast_2d(lr.coef_), dtype=torch.float32,
                                            device=dev))
        model.head.bias.copy_(torch.tensor(np.atleast_1d(lr.intercept_), dtype=torch.float32,
                                           device=dev))
    for p in model.head.parameters():
        p.requires_grad = True
    return float(best), bc


@torch.no_grad()
def predict(model, x, xyz, batch=8, binary=True):
    """Held-out predictions. Sigmoid only for a binary head; a ridge head is already the
    predicted value and squashing it would destroy the ordering Spearman reads."""
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = x[s:s + batch].to(DEVICE)
        o = model(xb, xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1))
        out.append((torch.sigmoid(o) if binary else o).cpu())
    return torch.cat(out).numpy()


def fit_all_and_save(args, x, y, xyz, man, binary, metric):
    """Fine-tune on every ADNI scan and write the encoder.

    Separate from the k-fold path on purpose. That path answers "does fine-tuning improve
    ADNI", which needs a held-out fold. This one answers "give me the encoder to carry to
    the RCTs", where ADNI is entirely training data and the RCT is the held-out set, so
    spending 1/3 of ADNI on an ADNI-side estimate would only weaken the checkpoint. It
    therefore prints no delta -- there is no honest one to print -- and the ADNI-side
    effect is read off the matching k-fold run instead.
    """
    seed = args.seeds[0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    cut = max(4, int(0.2 * len(y)))
    va, tr = perm[:cut], perm[cut:]

    model = ProbeHeadClassifier(
        pretrained_backbone_path=args.pretrained, lora_r=args.lora_r,
        unfreeze_block=args.unfreeze_block, readout=args.readout,
        layer=args.layer, n_pca=args.n_pca).to(DEVICE).eval()
    Ftr = frozen_features(model, x[tr], xyz, args.batch_size)
    inner_auc, cbest = fit_probe_into_head(model, Ftr, y[tr], seed=seed)
    a0 = _score(y[va], predict(model, x[va], xyz, args.batch_size, binary))
    npar = sum(q.numel() for q in model.parameters() if q.requires_grad)

    crit = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
    opt = torch.optim.AdamW(
        [{"params": [q for q in model.brainlm.parameters() if q.requires_grad],
          "lr": args.fm_lr},
         {"params": list(model.head.parameters()), "lr": args.head_lr}], weight_decay=1e-2)
    tl = DataLoader(TensorDataset(x[tr], torch.tensor(y[tr], dtype=torch.float32)),
                    batch_size=args.batch_size, shuffle=True)
    best_va, best_state, stale, best_epoch, ep_run = -np.inf, None, 0, 0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb, xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1)), yb).backward()
            opt.step()
        a = _score(y[va], predict(model, x[va], xyz, args.batch_size, binary))
        ep_run = ep
        if np.isfinite(a) and a > best_va:
            best_va, stale, best_epoch = a, 0, ep
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  fit_all: {len(tr)} train / {len(va)} inner-val | trainable {npar:,} | "
          f"inner-val {metric} {a0:.4f} -> {best_va:.4f} | best_ep {best_epoch}/{ep_run}",
          flush=True)
    print("  the inner-val figures above are NOT a held-out estimate: early stopping "
          "selected on them.")

    # Save the ViT in the shape load_brainlm_forpretraining expects, so downstream scripts
    # take a checkpoint path and need no knowledge of LoRA. merge_and_unload folds the LoRA
    # deltas into the base weights; without it the adapter is silently dropped on reload.
    enc = model.brainlm
    try:
        enc = enc.merge_and_unload()
    except AttributeError:
        pass                              # no LoRA to merge (unfreeze_block-only runs)
    dst = Path(args.save_encoder)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in enc.state_dict().items()}, dst)
    meta = dst.with_suffix(".meta.json")
    json.dump({"args": vars(args), "inner_val_frozen": float(a0),
               "inner_val_finetuned": float(best_va), "best_epoch": int(best_epoch),
               "epochs_run": int(ep_run), "n_train": int(len(tr)), "n_inner_val": int(len(va)),
               "trainable": int(npar)}, open(meta, "w"), indent=1)
    print(f"wrote {dst}\nwrote {meta}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Probe-initialised CN/AD fine-tuning, nested CV")
    d = DERIVED_DIR / "adni"
    ap.add_argument("--tag", default="raw_DX_first_t140_off")
    ap.add_argument("--in_dir", default=str(d))
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--readout", default="emb_fc",
                    choices=("cls", "mean", "roi_var", "roi_norm", "emb_fc"))
    ap.add_argument("--layer", type=int, default=3, help="sweep optimum for emb_fc on the 13M")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--unfreeze_block", type=int, default=None,
                    help="index of an encoder block to unfreeze (default: none, LoRA only). "
                         "The old script always unfroze the LAST block, which the sweep "
                         "measures as the worst layer for this task.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fm_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-4)
    ap.add_argument("--n_pca", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_encoder", default=None,
                    help="write the fine-tuned BrainLM weights here. Intended with "
                         "--fit_all: the RCTs are the held-out validation set, so an "
                         "encoder meant for RCT transfer should see ALL of ADNI, and the "
                         "k-fold protocol above exists only to measure the ADNI-side delta.")
    ap.add_argument("--fit_all", action="store_true",
                    help="one run on every ADNI scan, with an inner split for early "
                         "stopping and no held-out fold. Reports no delta by construction; "
                         "use it only to produce a checkpoint.")
    args = ap.parse_args()
    C.configure_backends()

    p = Path(args.in_dir)
    x = torch.load(p / f"ADNI_x_{args.tag}.pt", map_location="cpu", weights_only=False).float()
    y = torch.load(p / f"ADNI_y_{args.tag}.pt", map_location="cpu", weights_only=False).numpy()
    man = pd.read_csv(p / f"ADNI_manifest_{args.tag}.csv")
    xyz = load_xyz(XYZ_PATH).to(DEVICE)
    binary = len(np.unique(y)) == 2
    metric = "AUC" if binary else "rho"
    print(f"{args.tag}: X={tuple(x.shape)} target={'binary' if binary else 'ordinal/continuous'}"
          f" ({metric})  positives={int(y.sum())}/{len(y)}  "
          f"readout={args.readout} layer={args.layer} lora_r={args.lora_r} "
          f"unfreeze_block={args.unfreeze_block}", flush=True)

    if args.fit_all:
        fit_all_and_save(args, x, y, xyz, man, binary, metric)
        return

    per_seed_ep0, per_seed_fin, rows = [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = man.copy()
        # assign_folds stratifies on a binary `label`; for a graded target use its strata.
        m["label"] = y.astype(int) if binary else _strata(y)
        folds, _ = assign_folds(m, n_splits=args.folds, seed=seed, verbose=False)
        oof0 = np.full(len(y), np.nan)
        oofF = np.full(len(y), np.nan)
        for k in sorted(set(folds)):
            te = folds == k
            tr_all = np.where(~te)[0]
            rng = np.random.default_rng(seed * 100 + k)
            perm = rng.permutation(len(tr_all))
            cut = max(4, int(0.2 * len(tr_all)))
            va, tr = tr_all[perm[:cut]], tr_all[perm[cut:]]

            model = ProbeHeadClassifier(
                pretrained_backbone_path=args.pretrained, lora_r=args.lora_r,
                unfreeze_block=args.unfreeze_block, readout=args.readout,
                layer=args.layer, n_pca=args.n_pca).to(DEVICE).eval()
            Ftr = frozen_features(model, x[tr], xyz, args.batch_size)
            inner_auc, cbest = fit_probe_into_head(model, Ftr, y[tr], seed=seed)
            oof0[te] = predict(model, x[te], xyz, args.batch_size, binary)
            a0 = _score(y[te], oof0[te]) if len(set(y[te])) > 1 else float("nan")

            npar = sum(q.numel() for q in model.parameters() if q.requires_grad)
            if binary:
                pos = float((1 - y[tr]).sum() / max(y[tr].sum(), 1))
                crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos], device=DEVICE))
            else:
                crit = nn.MSELoss()
            opt = torch.optim.AdamW(
                [{"params": [q for q in model.brainlm.parameters() if q.requires_grad],
                  "lr": args.fm_lr},
                 {"params": list(model.head.parameters()), "lr": args.head_lr}],
                weight_decay=1e-2)
            tl = DataLoader(TensorDataset(x[tr], torch.tensor(y[tr], dtype=torch.float32)),
                            batch_size=args.batch_size, shuffle=True)
            best_va, best_state, stale = -np.inf, None, 0
            # Instrumented because "+0.002 downstream" has two very different causes and
            # the artefact could not tell them apart: an encoder that moved and did not
            # help, or an encoder that never moved. `best_epoch` says whether early
            # stopping fired immediately, `train_auc` says whether the model can fit the
            # training set at all, and `w_shift` is the relative L2 change of the LoRA
            # parameters -- if that is ~0 then fm_lr was simply too small and the result is
            # an under-training result, not a capacity result.
            w0 = {kk: v.detach().clone() for kk, v in model.state_dict().items()
                  if v.dtype.is_floating_point}
            best_epoch, ep_run = 0, 0
            for ep in range(1, args.epochs + 1):
                model.train()
                for xb, yb in tl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    crit(model(xb, xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1)), yb).backward()
                    opt.step()
                pv = predict(model, x[va], xyz, args.batch_size, binary)
                a = _score(y[va], pv) if len(set(y[va])) > 1 else float("nan")
                ep_run = ep
                if np.isfinite(a) and a > best_va:
                    best_va, stale, best_epoch = a, 0, ep
                    best_state = {kk: v.detach().cpu().clone()
                                  for kk, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            num = den = 0.0
            for kk, v in model.state_dict().items():
                if kk in w0:
                    num += float((v.detach().cpu() - w0[kk].cpu()).pow(2).sum())
                    den += float(w0[kk].cpu().pow(2).sum())
            w_shift = (num / max(den, 1e-12)) ** 0.5
            ptr = predict(model, x[tr], xyz, args.batch_size, binary)
            train_auc = _score(y[tr], ptr) if len(set(y[tr])) > 1 else float("nan")
            oofF[te] = predict(model, x[te], xyz, args.batch_size, binary)
            aF = _score(y[te], oofF[te]) if len(set(y[te])) > 1 else float("nan")
            print(f"  seed {seed} fold {k}: {len(tr)} train / {len(va)} inner-val / "
                  f"{int(te.sum())} held out | trainable {npar:,} | "
                  f"epoch-0 (frozen probe) {a0:.4f} -> fine-tuned {aF:.4f} | "
                  f"best_ep {best_epoch}/{ep_run}  train {train_auc:.4f}  "
                  f"|dW|/|W| {w_shift:.2e}", flush=True)
            rows.append({"seed": seed, "fold": int(k), "auc_epoch0": a0, "auc_final": aF,
                         "inner_val_auc": inner_auc, "C": cbest, "trainable": npar,
                         "best_epoch": int(best_epoch), "epochs_run": int(ep_run),
                         "train_auc": float(train_auc), "w_shift": float(w_shift)})
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        v0 = np.isfinite(oof0)
        per_seed_ep0.append(_score(y[v0], oof0[v0]))
        vF = np.isfinite(oofF)
        per_seed_fin.append(_score(y[vF], oofF[vF]))
        print(f"seed {seed}: pooled out-of-fold  frozen probe {per_seed_ep0[-1]:.4f}  "
              f"fine-tuned {per_seed_fin[-1]:.4f}", flush=True)

    print(f"\n{'=' * 78}")
    print(f"{args.tag}  readout={args.readout} layer={args.layer} "
          f"lora_r={args.lora_r} unfreeze_block={args.unfreeze_block}")
    print(f"  frozen probe (epoch 0)  {metric} {np.mean(per_seed_ep0):.4f} +/- {np.std(per_seed_ep0):.4f}")
    print(f"  after fine-tuning       {metric} {np.mean(per_seed_fin):.4f} +/- {np.std(per_seed_fin):.4f}")
    print(f"  delta                   {np.mean(per_seed_fin) - np.mean(per_seed_ep0):+.4f}")
    print("  both numbers are pooled out-of-fold under the same nested protocol, so they are")
    print("  directly comparable -- unlike the old script's val-split maximum.")
    print("=" * 78)

    out = Path(args.out) if args.out else (RUNS_DIR / "v2" / "finetune_probe" /
                                           f"{args.tag}_{args.readout}_L{args.layer}"
                                           f"_r{args.lora_r}_ub{args.unfreeze_block}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "frozen_probe_auc": [float(np.mean(per_seed_ep0)), float(np.std(per_seed_ep0))],
               "finetuned_auc": [float(np.mean(per_seed_fin)), float(np.std(per_seed_fin))],
               "per_fold": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
