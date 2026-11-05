"""Shared evaluation protocol for every model in the benchmark.

The original pipeline trained for a fixed 20 epochs, scored the validation fold after
each one, and reported the *best* epoch's validation metrics as the fold result. The
same data therefore selected the model and produced the number, with no held-out set
anywhere. Measured on the shipped reference logs, that inflates accuracy by 0.06-0.12
-- enough that none of the six settings beats its majority-class baseline once the
selection is removed.

This module supplies the honest alternative and, just as importantly, makes every
model use it. Under the old code the foundation model got best-of-20-epochs on the
scored fold while the SVM/MLP/GCN baselines got a single final model, so the Q1
comparison measured protocol, not architecture.

Protocol
--------
    for each seed:
      for each outer fold k (from the stratified fold column in the label CSV):
        inner_train, inner_val = stratified split of the other four folds
        train on class-balanced inner_train, choose the epoch by inner_val AUC
        score fold k exactly once with that epoch's weights

Reported: per-fold metrics, their mean +/- sd, and pooled out-of-fold metrics
(predictions from all five folds concatenated, then scored once). At n=15-19 per
fold the pooled figure is the more stable estimate; the mean-of-folds is kept
because it is what the manuscript reports.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import config as C

METRIC_NAMES = ("acc", "bal_acc", "auc", "f1", "sens", "spec")


def logits(model, x):
    """Call a head and return shape [B]. Accepts heads that emit [B] or [B, 1]."""
    out = model(x)
    return out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out


def compute_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    """Metrics from probabilities. AUC is nan for a single-class fold rather than
    an exception, so one degenerate fold does not abort a run."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs > threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y_true, preds),
        "bal_acc": balanced_accuracy_score(y_true, preds),
        "auc": auc,
        "f1": f1_score(y_true, preds, zero_division=0),
        "sens": tp / (tp + fn) if (tp + fn) else 0.0,
        "spec": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def majority_baseline(y_train, y_val) -> dict:
    """Always predict the training-set majority class. Any model that does not beat
    this on accuracy has not learned anything, which is easy to miss on CogTE where
    the majority class is 76% of the cohort."""
    maj = int(np.round(np.mean(np.asarray(y_train)))) if len(y_train) else 0
    probs = np.full(len(y_val), float(maj))
    return compute_metrics(y_val, probs)


def balance_indices(idx, labels, seed: int):
    """Oversample the minority class of a training split up to parity."""
    idx = np.asarray(idx)
    lab = np.asarray(labels)[idx]
    i0, i1 = idx[lab == 0], idx[lab == 1]
    n = max(len(i0), len(i1))
    rng = np.random.default_rng(seed)
    if 0 < len(i0) < n:
        i0 = rng.choice(i0, n, replace=True)
    if 0 < len(i1) < n:
        i1 = rng.choice(i1, n, replace=True)
    out = np.concatenate([i0, i1])
    rng.shuffle(out)
    return out


def inner_split(train_idx, labels, seed: int, val_frac: float = None):
    """Stratified inner split of an outer training fold, used only to pick the epoch."""
    val_frac = C.INNER_VAL_FRAC if val_frac is None else val_frac
    train_idx = np.asarray(train_idx)
    lab = np.asarray(labels)[train_idx]
    # Fall back to no inner split if a class is too rare to stratify.
    if min(np.bincount(lab.astype(int), minlength=2)) < 2:
        return train_idx, train_idx
    inner_tr, inner_va = train_test_split(
        train_idx, test_size=val_frac, random_state=seed, stratify=lab
    )
    return inner_tr, inner_va


@dataclass
class FoldResult:
    fold: int
    seed: int
    metrics: dict
    baseline: dict
    best_epoch: int
    inner_auc: float
    val_idx: np.ndarray
    val_true: np.ndarray
    val_prob: np.ndarray
    head_state: dict = field(default=None, repr=False)


def train_head_nested(
    features: torch.Tensor,
    labels: np.ndarray,
    train_idx,
    val_idx,
    build_model,
    seed: int,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    device=None,
    fold: int = 0,
    logger=None,
):
    """Train one head with inner-validation epoch selection; score `val_idx` once.

    `build_model()` must return a fresh nn.Module mapping [B, D] -> [B] logits.
    """
    epochs = C.EPOCHS if epochs is None else epochs
    lr = C.LEARNING_RATE if lr is None else lr
    batch_size = C.BATCH_SIZE if batch_size is None else batch_size
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    inner_tr, inner_va = inner_split(train_idx, labels, seed)
    inner_tr_bal = balance_indices(inner_tr, labels, seed)

    feats = features.to(device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    gen = torch.Generator().manual_seed(seed)
    best_auc, best_state, best_epoch = -np.inf, copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(inner_tr_bal), generator=gen).numpy()
        shuffled = inner_tr_bal[order]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            optimizer.zero_grad()
            loss = criterion(logits(model, feats[batch]), y[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            inner_prob = torch.sigmoid(logits(model, feats[inner_va])).cpu().numpy()
        inner_metrics = compute_metrics(labels[inner_va], inner_prob)
        # Selection criterion is inner-validation AUC: threshold-free, and unlike
        # accuracy it is not maximised by collapsing onto the majority class.
        score = inner_metrics["auc"]
        if not np.isnan(score) and score > best_auc:
            best_auc, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if logger:
            logger.info(
                f"  fold {fold} seed {seed} epoch {epoch:>2}: "
                f"inner_auc={score:.4f} inner_acc={inner_metrics['acc']:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(logits(model, feats[val_idx])).cpu().numpy()

    return FoldResult(
        fold=fold,
        seed=seed,
        metrics=compute_metrics(labels[val_idx], val_prob),
        baseline=majority_baseline(labels[train_idx], labels[val_idx]),
        best_epoch=best_epoch,
        inner_auc=best_auc,
        val_idx=np.asarray(val_idx),
        val_true=np.asarray(labels)[val_idx],
        val_prob=val_prob,
        head_state={k: v.detach().cpu().clone() for k, v in best_state.items()},
    )


def summarize(results: list[FoldResult], logger=None, title: str = "") -> dict:
    """Aggregate fold results: mean +/- sd across folds, and pooled out-of-fold."""
    by_seed: dict[int, list[FoldResult]] = {}
    for r in results:
        by_seed.setdefault(r.seed, []).append(r)

    def log(msg):
        print(msg, flush=True)
        if logger:
            logger.info(msg)

    seed_rows, pooled_rows = {}, {}
    for seed, rows in sorted(by_seed.items()):
        seed_rows[seed] = {m: np.array([r.metrics[m] for r in rows], dtype=float) for m in METRIC_NAMES}
        order = np.argsort(np.concatenate([r.val_idx for r in rows]))
        true = np.concatenate([r.val_true for r in rows])[order]
        prob = np.concatenate([r.val_prob for r in rows])[order]
        pooled_rows[seed] = compute_metrics(true, prob)

    base = {m: np.array([r.baseline[m] for r in results], dtype=float) for m in METRIC_NAMES}

    if title:
        log("\n" + "=" * 76)
        log(title)
        log("=" * 76)

    log("\nPer-seed mean +/- sd across the 5 outer folds (each fold scored once):")
    for seed in sorted(seed_rows):
        vals = seed_rows[seed]
        log(
            f"  seed {seed}: "
            + "  ".join(f"{m}={np.nanmean(vals[m]):.4f}+/-{np.nanstd(vals[m]):.4f}" for m in METRIC_NAMES)
        )

    log("\nPooled out-of-fold (all folds' predictions scored together):")
    for seed in sorted(pooled_rows):
        log("  seed %d: " % seed + "  ".join(f"{m}={pooled_rows[seed][m]:.4f}" for m in METRIC_NAMES))

    across = {m: np.array([np.nanmean(seed_rows[s][m]) for s in seed_rows]) for m in METRIC_NAMES}
    pooled_across = {m: np.array([pooled_rows[s][m] for s in pooled_rows]) for m in METRIC_NAMES}

    log("\nAcross seeds (mean +/- sd of the per-seed values above):")
    log("  fold-mean : " + "  ".join(f"{m}={np.nanmean(across[m]):.4f}+/-{np.nanstd(across[m]):.4f}" for m in METRIC_NAMES))
    log("  pooled OOF: " + "  ".join(f"{m}={np.nanmean(pooled_across[m]):.4f}+/-{np.nanstd(pooled_across[m]):.4f}" for m in METRIC_NAMES))

    log("\nMajority-class baseline (predict the training-fold majority):")
    log("  " + "  ".join(f"{m}={np.nanmean(base[m]):.4f}" for m in METRIC_NAMES))

    delta = np.nanmean(across["acc"]) - np.nanmean(base["acc"])
    log(f"\nAccuracy above majority baseline: {delta:+.4f}")
    log("Selected epochs: " + ", ".join(str(r.best_epoch) for r in results))

    return {
        "per_seed_fold_mean": {s: {m: float(np.nanmean(v[m])) for m in METRIC_NAMES} for s, v in seed_rows.items()},
        "per_seed_pooled": {s: {m: float(v[m]) for m in METRIC_NAMES} for s, v in pooled_rows.items()},
        "across_seeds_fold_mean": {m: [float(np.nanmean(across[m])), float(np.nanstd(across[m]))] for m in METRIC_NAMES},
        "across_seeds_pooled": {m: [float(np.nanmean(pooled_across[m])), float(np.nanstd(pooled_across[m]))] for m in METRIC_NAMES},
        "majority_baseline": {m: float(np.nanmean(base[m])) for m in METRIC_NAMES},
    }
