#!/usr/bin/env python3
"""Domain-adaptive continued pretraining of BrainLM on ADNI, with no labels.

Why this instead of more supervised fine-tuning. Measured on this data:

  * the AD/HC target the stage-1 fine-tune uses is not learnable from these scans -- the
    imaging model reaches pooled OOF AUC 0.4979 against a 0.5000 majority line, while the
    acquisition protocol alone reaches 0.6219 and four demographic variables reach 0.7586;
  * fine-tuning on a target that *is* present in fMRI (sex) makes the representation
    *worse*, 0.5941 frozen -> 0.5215 fine-tuned, with inner-validation scores peaking at
    epoch 1-7 and uncorrelated with held-out scores. 1.7 M trainable parameters against
    326 training subjects is fitting noise.

Continuing the model's own masked-reconstruction objective avoids both problems: it needs
no labels, so every session is usable rather than only the labelled ones, and the loss is
dense -- one residual per (ROI, time-patch) instead of one bit per subject.

Reference practice: domain-adaptive pretraining, i.e. continue the self-supervised
objective on in-domain text/data before any task fine-tuning (Gururangan et al., 2020,
"Don't Stop Pretraining"). The same argument applies here with masked reconstruction in
place of masked language modelling.

**The RCT cohorts are deliberately excluded.** ACT, CogTE and BREATHE stay untouched so
they remain fully independent validation sets. Nothing in this script reads `data/rct/`.
That is stricter than test-time adaptation would allow, and it is the right default: any
downstream number is then a clean out-of-domain generalisation result.

Split: grouped by subject, so no subject appears in both train and validation. Windows are
sampled at random start positions each epoch, which is the natural augmentation here -- a
976-frame recording supplies many distinct 140-frame windows.

Example:
  python scripts/dapt_adni.py --epochs 30
  python scripts/dapt_adni.py --window 200 --epochs 30 --tag w200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from datasets import load_xyz  # noqa: E402
from model_loader import load_brainlm_forpretraining  # noqa: E402
from paths import ADNI_SCAN_DIR, OFFICIAL_BRAINLM_13M, RUNS_DIR, XYZ_PATH  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBJECT_RE = re.compile(r"(sub-ADNI\d+S\d+)_")


class AdniWindows(Dataset):
    """Random `window`-frame crops from per-session [424, T] scans.

    Scans are held in memory as float32; the 1617 ADNI sessions at 140-976 frames are a
    few GB, which fits. A fresh random start per __getitem__ means the effective sample
    count is far larger than the session count, and it is the only augmentation used --
    anything that alters the signal would defeat the purpose of adapting to its
    distribution.
    """

    def __init__(self, files: list[Path], window: int, seed: int = 0, deterministic: bool = False):
        self.window = window
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)
        self.scans, self.kept = [], []
        for f in files:
            s = torch.load(f, map_location="cpu", weights_only=False).float()
            if s.shape[0] != C.NUM_BRAIN_VOXELS or s.shape[1] < window:
                continue
            # A zero-variance parcel carries no signal and would dominate an MSE loss
            # with a trivially predictable constant.
            g = s.std(dim=1, unbiased=False)
            live = g[g > 1e-6]
            # Off-scale scans: 4% of the release sits 2-3 orders of magnitude from the
            # cohort median and its amplitude profile correlates with the same subject's
            # other scan BELOW the between-subject baseline, i.e. it is corrupt rather than
            # merely rescaled (data/README.md). An MSE reconstruction loss would be
            # dominated by those scans.
            if len(live) and not (0.05 < float(live.median()) < 20.0):
                continue
            if (s.std(dim=1) < 1e-6).any():
                continue
            self.scans.append(s)
            self.kept.append(f)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        s = self.scans[i]
        span = s.shape[1] - self.window
        start = span // 2 if self.deterministic else int(self.rng.integers(0, span + 1))
        w = s[:, start:start + self.window]
        # Mean-centre per ROI, then divide the WHOLE window by ONE scalar (its median
        # per-ROI sd). This matches `prepare_adni.py --amplitude preserve` exactly, and the
        # distinction matters: the previous version divided by each ROI's own sd, which sets
        # every per-ROI sd to 1 and destroys between-ROI amplitude. That was correct for the
        # deleted `denoise-old` release, whose scans were already z-scored, but on the
        # amplitude-bearing `raw` release it silently threw away the very thing BrainLM's
        # pretraining distribution has and this adaptation is supposed to learn -- so DAPT
        # on `raw` would have reproduced DAPT on `denoise-old`.
        w = w - w.mean(dim=1, keepdim=True)
        gain = w.std(dim=1, unbiased=False).median().clamp_min(1e-6)
        return w / gain


def subject_split(files: list[Path], val_frac: float, seed: int):
    subs = sorted({SUBJECT_RE.match(f.name).group(1) for f in files})
    rng = np.random.default_rng(seed)
    val = set(np.array(subs)[rng.permutation(len(subs))[: max(1, int(val_frac * len(subs)))]])
    tr = [f for f in files if SUBJECT_RE.match(f.name).group(1) not in val]
    va = [f for f in files if SUBJECT_RE.match(f.name).group(1) in val]
    return tr, va, len(subs), len(val)


class masked_eval:
    """Context for measuring reconstruction loss without training-mode randomness.

    `BrainLMEmbeddings.forward` skips masking entirely in eval mode -- correct for CLS
    extraction, but it returns an all-zero mask, and `forward_loss` divides by
    `mask.sum()`, so the reconstruction loss is 0/0 = NaN. Masking therefore has to stay
    on, which means train mode. Train mode also switches dropout on
    (attention_probs_dropout_prob = 0.1 in the released config), which would make the
    validation loss noisy for a reason unrelated to the model, so the dropout
    probabilities are zeroed for the duration and restored afterwards.
    """

    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                self.saved.append((m, m.p))
                m.p = 0.0
        self.was_training = self.model.training
        self.model.train()
        return self.model

    def __exit__(self, *exc):
        for m, p in self.saved:
            m.p = p
        self.model.train(self.was_training)
        return False


@torch.no_grad()
def evaluate(model, loader, xyz):
    """Validation reconstruction loss with the mask pattern held fixed across epochs.

    `noise` is passed explicitly so every epoch masks the same patches; otherwise the
    epoch-to-epoch change in validation loss is dominated by which patches happened to be
    hidden rather than by the model improving. Its shape must be [batch, seq_length] with
    seq_length = num_voxels * patches_per_voxel -- the flattened token sequence, not the
    2D (voxel, patch) grid.
    """
    cfg = model.config
    seq_len = cfg.num_brain_voxels * (cfg.num_timepoints_per_voxel // cfg.timepoint_patching_size)
    tot, n = 0.0, 0
    with masked_eval(model):
        for i, xb in enumerate(loader):
            xb = xb.to(DEVICE)
            g = torch.Generator().manual_seed(1234 + i)
            noise = torch.rand(xb.shape[0], seq_len, generator=g).to(DEVICE)
            out = model(signal_vectors=xb,
                        xyz_vectors=xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1), noise=noise)
            loss = out.loss.detach()
            if not torch.isfinite(loss):
                raise RuntimeError(f"validation loss is {loss.item()} on batch {i}")
            tot += loss.item() * xb.shape[0]
            n += xb.shape[0]
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Domain-adaptive pretraining on ADNI (no labels)")
    ap.add_argument("--scan_dir", default=str(ADNI_SCAN_DIR))
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--window", type=int, default=140,
                    help="frames per crop; must be a multiple of the 20-frame patch. 140 "
                         "keeps every ADNI session (7 patches/ROI); 200 matches BrainLM's "
                         "pretraining window but drops the 140-frame sessions.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="deliberately small: this continues an existing optimum rather "
                         "than starting one")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="dapt")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    C.configure_backends()

    if args.window % C.TIMEPOINT_PATCHING_SIZE:
        raise SystemExit(f"--window must be a multiple of {C.TIMEPOINT_PATCHING_SIZE}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(Path(args.scan_dir).glob("*.pt"))
    if not files:
        raise SystemExit(f"no .pt scans under {args.scan_dir}")
    tr_f, va_f, n_sub, n_val_sub = subject_split(files, args.val_frac, args.seed)
    print(f"{len(files)} sessions / {n_sub} subjects from {args.scan_dir}")
    print(f"  grouped split: {len(tr_f)} train / {len(va_f)} val sessions "
          f"({n_sub - n_val_sub} / {n_val_sub} subjects, no overlap)", flush=True)

    tr_ds = AdniWindows(tr_f, args.window, seed=args.seed)
    va_ds = AdniWindows(va_f, args.window, seed=args.seed + 1, deterministic=True)
    print(f"  usable after QC and length filter: {len(tr_ds)} train / {len(va_ds)} val", flush=True)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size)

    # The window sets num_timepoints_per_voxel, so the model must be built for it.
    C.NUM_TIMEPOINTS_PER_VOXEL = args.window
    model = load_brainlm_forpretraining(args.pretrained).to(DEVICE)
    for p in model.parameters():          # load_brainlm_forpretraining freezes everything
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  window {args.window} -> {args.window // C.TIMEPOINT_PATCHING_SIZE} patches/ROI, "
          f"mask_ratio {model.config.mask_ratio}, {n_train / 1e6:.2f} M trainable", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * max(1, len(tr_dl))
    warm = int(args.warmup_frac * total)

    def lr_at(step):
        if step < warm:
            return step / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    xyz = load_xyz(XYZ_PATH).to(DEVICE)

    out_dir = Path(args.out_dir) if args.out_dir else RUNS_DIR / "v2" / "dapt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{args.tag}_w{args.window}.pth"

    base_val = evaluate(model, va_dl, xyz)
    print(f"\nepoch  0 (official weights, no adaptation): val recon MSE {base_val:.6f}", flush=True)
    hist = [{"epoch": 0, "train": None, "val": base_val}]
    best, best_ep, stale = base_val, 0, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for xb in tr_dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            out = model(signal_vectors=xb,
                        xyz_vectors=xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += out.loss.detach().item()
            nb += 1
        val = evaluate(model, va_dl, xyz)
        hist.append({"epoch": ep, "train": run / max(nb, 1), "val": val})
        flag = ""
        if val < best:
            best, best_ep, stale = val, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "val_recon_mse": val, "window": args.window,
                        "base_val_recon_mse": base_val, "scan_dir": str(args.scan_dir)}, ckpt)
            flag = "  *saved"
        else:
            stale += 1
        print(f"epoch {ep:2d}: train {run / max(nb, 1):.6f}  val {val:.6f}{flag}", flush=True)
        if stale >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs", flush=True)
            break

    print(f"\n{'=' * 70}")
    print(f"best epoch {best_ep}: val recon MSE {best:.6f}  "
          f"(official weights {base_val:.6f}, {100 * (base_val - best) / base_val:+.1f}%)")
    print(f"{'=' * 70}")
    if best_ep == 0:
        print("The official weights were never beaten: continued pretraining did not fit\n"
              "this data better than the released model already does. Nothing was saved.")
    else:
        print(f"checkpoint -> {ckpt}")
        print("Use it downstream with --pretrained, and note it saw ADNI only: ACT, CogTE\n"
              "and BREATHE were not touched, so they remain independent validation sets.")
    (out_dir / f"{args.tag}_w{args.window}_history.json").write_text(
        json.dumps({"args": vars(args), "base_val": base_val, "best_epoch": best_ep,
                    "best_val": best, "history": hist}, indent=2))


if __name__ == "__main__":
    main()
