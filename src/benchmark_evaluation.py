"""Shared, subject-grouped nested probes and paired conditional uncertainty.

Scores retain session weighting when subjects have multiple visits. Confidence
intervals resample whole subjects and condition on the fitted OOF predictions;
they do not include uncertainty from refitting the encoder or selecting a recipe.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROTOCOL_VERSION = "aging-benchmark-2026-09-23-v1"
GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def strata(y):
    y = np.asarray(y)
    levels = np.unique(y)
    if len(levels) <= 10:
        return np.searchsorted(levels, y)
    return np.searchsorted(np.quantile(y, [.2, .4, .6, .8]), y)


def grouped_splits(y, groups, seed, n_splits=5):
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups) or not np.isfinite(y).all():
        raise ValueError("Finite labels and one subject group per row are required")
    k = min(n_splits, len(np.unique(groups)))
    if k < 2:
        raise ValueError("At least two subjects are required")
    cv = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    for tr, te in cv.split(np.zeros(len(y)), strata(y), groups):
        if set(groups[tr]) & set(groups[te]):
            raise RuntimeError("Subject leakage across folds")
        yield tr, te


def transform_blocks(blocks, ks, tr, te):
    """All learned transforms are fitted only on the supplied training indices."""
    train, test = [], []
    if len(blocks) != len(ks):
        raise ValueError("One component count is required per feature block")
    for block, k in zip(blocks, ks):
        block = np.asarray(block)
        sc = StandardScaler().fit(block[tr])
        a, b = sc.transform(block[tr]), sc.transform(block[te])
        # Full-width blocks (e.g. clinical covariates) need no rotation.
        if k < block.shape[1]:
            pc = PCA(n_components=min(k, len(tr) - 1), random_state=0).fit(a)
            a, b = pc.transform(a), pc.transform(b)
        train.append(a)
        test.append(b)
    return np.hstack(train), np.hstack(test)


def prepare_nested(blocks, ks, y, groups, seed, n_splits=5):
    groups = np.asarray(groups)
    prepared = []
    for tr, te in grouped_splits(y, groups, seed, n_splits):
        if len(np.unique(groups[tr])) < 2:
            raise ValueError("Outer training fold needs at least two subjects for tuning")
        it, iv = next(GroupShuffleSplit(1, test_size=.2, random_state=seed)
                      .split(np.zeros(len(tr)), groups=groups[tr]))
        itr, iva = tr[it], tr[iv]
        prepared.append((tr, te, transform_blocks(blocks, ks, tr, te),
                         itr, iva, transform_blocks(blocks, ks, itr, iva)))
    return prepared


def _predict(a, y, b, c, binary):
    if binary and len(np.unique(y)) < 2:
        return np.full(len(b), float(np.mean(y)))
    model = (LogisticRegression(C=c, max_iter=5000) if binary
             else Ridge(alpha=1.0 / c)).fit(a, y)
    return model.predict_proba(b)[:, 1] if binary else model.predict(b)


def score(y, pred, binary=None):
    y, pred = np.asarray(y), np.asarray(pred)
    if binary is None:
        binary = set(np.unique(y)) == {0, 1}
    if binary:
        return float(roc_auc_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    if np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(stats.spearmanr(y, pred).statistic)


def predict_prepared(prepared, y, grid=GRID, binary=None):
    """Refit every supervised step; safe to reuse transforms for label-independent splits."""
    y = np.asarray(y)
    if binary is None:
        binary = set(np.unique(y)) == {0, 1}
    oof = np.full(len(y), np.nan)
    for tr, te, (a, b), itr, iva, (ai, bi) in prepared:
        best, best_score = grid[0], -np.inf
        for c in grid:
            s = score(y[iva], _predict(ai, y[itr], bi, c, binary), binary)
            if np.isfinite(s) and s > best_score:
                best, best_score = c, s
        oof[te] = _predict(a, y[tr], b, best, binary)
    if not np.isfinite(oof).all():
        raise ValueError("Non-finite or missing out-of-fold predictions")
    return oof


def oof_predictions(blocks, k_per_block, y, seed, grid=GRID, groups=None, n_splits=5):
    y = np.asarray(y)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    prep = prepare_nested(blocks, k_per_block, y, groups, seed, n_splits)
    return predict_prepared(prep, y, grid)


def paired_bootstrap(y, pred_a, pred_b, n_boot=10000, seed=0, groups=None,
                     return_details=False):
    """Subject-cluster bootstrap, conditional on fitted OOF models; b minus a.

    The one-sided bootstrap tail is reported as such, not an exact permutation p.
    Invalid draws (e.g. one class) are excluded and their count is recorded.
    """
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(groups) != len(y) or n_boot < 2:
        raise ValueError("Require aligned groups and at least two bootstrap draws")
    binary = set(np.unique(y)) == {0, 1}
    obs = score(y, pred_b, binary) - score(y, pred_a, binary)
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in subjects}
    diffs = []
    for _ in range(n_boot):
        ix = np.concatenate([indices[g] for g in rng.choice(subjects, len(subjects), replace=True)])
        d = score(y[ix], pred_b[ix], binary) - score(y[ix], pred_a[ix], binary)
        if np.isfinite(d):
            diffs.append(d)
    if len(diffs) < 2:
        raise ValueError("Too few valid bootstrap draws")
    diffs = np.asarray(diffs)
    result = (float(obs), float((1 + np.sum(diffs <= 0)) / (len(diffs) + 1)),
              tuple(map(float, np.percentile(diffs, [2.5, 97.5]))))
    details = {"bootstrap_se": float(diffs.std(ddof=1)),
               "bootstrap_valid": len(diffs), "bootstrap_requested": n_boot,
               "n_subjects": len(subjects), "uncertainty": "conditional_on_fitted_oof",
               "resampling_unit": "subject", "score_weighting": "session"}
    return (*result, details) if return_details else result


def permutation_tail(observed, null, two_sided=False, center=0.0):
    null = np.asarray(null, dtype=float)
    if len(null) == 0 or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("Permutation statistic and all null draws must be finite")
    if two_sided:
        exceed = np.abs(null - center) >= abs(observed - center)
    else:
        exceed = null >= observed
    return float((1 + exceed.sum()) / (len(null) + 1))
