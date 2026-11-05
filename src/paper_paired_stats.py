"""Conditional paired inference, with participant-cluster resampling.

Bootstrap ranks are recomputed with multiplicities, not bootstrapped fixed ranks.
Permutation swaps entire participants' predictions between methods. This tests
conditional prediction exchangeability, not a null obtained by refitting models.
"""
import numpy as np
from scipy.stats import rankdata


def score_batch(y, predictions, binary=False):
    y = np.asarray(y, float)
    p = np.atleast_2d(np.asarray(predictions, float))
    ranks = rankdata(p, axis=1)
    if binary:
        pos = y == 1
        n1, n0 = pos.sum(), (~pos).sum()
        if not n1 or not n0:
            return np.full(len(p), np.nan)
        return (ranks[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    r = rankdata(y)
    r -= r.mean()
    ranks -= ranks.mean(1, keepdims=True)
    denominator = np.sqrt((r*r).sum() * (ranks*ranks).sum(1))
    return np.divide((ranks*r).sum(1), denominator, out=np.full(len(p), np.nan), where=denominator > 0)


def weighted_ranks(values, weights):
    _, inv = np.unique(values, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    start = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
    counts = np.add.reduceat(weights[:, order], start, axis=1)
    ranks = np.cumsum(counts, axis=1) - (counts - 1) / 2
    return ranks[:, inv]


def weighted_scores(y, predictions, weights, binary=False):
    p = weighted_ranks(predictions, weights)
    if binary:
        positive = y == 1
        n1 = weights[:, positive].sum(1)
        n0 = weights[:, ~positive].sum(1)
        numer = (p[:, positive]*weights[:, positive]).sum(1) - n1*(n1+1)/2
        return np.divide(numer, n1*n0, out=np.full(len(weights), np.nan), where=n1*n0 > 0)
    r = weighted_ranks(y, weights)
    total = weights.sum(1, keepdims=True)
    p -= (p*weights).sum(1, keepdims=True)/total
    r -= (r*weights).sum(1, keepdims=True)/total
    den = np.sqrt((weights*p*p).sum(1)*(weights*r*r).sum(1))
    return np.divide((weights*p*r).sum(1), den, out=np.full(len(weights), np.nan), where=den > 0)


def paired(y, reference, candidate, groups, metric='rho', n_boot=5000, n_perm=20000, seed=0):
    y, a, b = [np.asarray(x, float) for x in (y, reference, candidate)]
    groups = np.asarray(groups).astype(str)
    if not (y.ndim == a.ndim == b.ndim == groups.ndim == 1 and len(y) == len(a) == len(b) == len(groups)):
        raise ValueError('Unaligned paired arrays')
    if not np.isfinite(np.r_[y, a, b]).all():
        raise ValueError('Nonfinite prediction or outcome')
    if metric not in ('rho', 'auc') or n_boot < 2 or n_perm < 1:
        raise ValueError('Unsupported metric or resampling budget')
    binary = metric == 'auc'
    if binary and not set(np.unique(y)) <= {0, 1}:
        raise ValueError('AUC requires binary outcomes')
    subjects, inverse = np.unique(groups, return_inverse=True)
    if len(subjects) < 2:
        raise ValueError('Need at least two participants')
    sa, sb = score_batch(y, np.stack([a, b]), binary)
    observed = float(sb-sa)
    result = dict(reference_score=float(sa), candidate_score=float(sb), delta=observed,
                  n=len(y), n_subjects=len(subjects), metric=metric,
                  uncertainty='Conditional on fitted OOF predictions; session-weighted statistic, participant-cluster resampling',
                  test='Two-sided participant-cluster prediction-swap permutation; conditional exchangeability; no refitting',
                  bootstrap_requested=n_boot, permutations_requested=n_perm, resampling_seed=seed)
    if not np.isfinite(observed):
        return dict(result, estimable=False, ci95=None, permutation_p=None, reason='Constant prediction or outcome')
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n_boot, 128):
        count = min(128, n_boot-start)
        weights = rng.multinomial(len(subjects), np.full(len(subjects), 1/len(subjects)), size=count)[:, inverse]
        draws.extend(weighted_scores(y,b,weights,binary)-weighted_scores(y,a,weights,binary))
    draws = np.asarray(draws)
    draws = draws[np.isfinite(draws)]
    if len(draws) < .95*n_boot:
        raise ValueError('More than 5% bootstrap replicates not estimable')
    exceed = 0
    valid = 0
    for start in range(0, n_perm, 128):
        mask = (rng.random((min(128,n_perm-start),len(subjects))) < .5)[:, inverse]
        pa, pb = np.where(mask,b,a), np.where(mask,a,b)
        null = score_batch(y,pb,binary)-score_batch(y,pa,binary)
        if not np.isfinite(null).all():
            raise ValueError('Nonestimable prediction-swap draw')
        exceed += int(np.sum(np.abs(null) >= abs(observed)-1e-12))
        valid += len(null)
    return dict(result, estimable=True, ci95=np.quantile(draws,[.025,.975]).tolist(),
                bootstrap_valid=len(draws), bootstrap_se=float(draws.std(ddof=1)),
                permutations_valid=valid, permutation_exceedances=exceed,
                permutation_p=(1+exceed)/(1+valid))


def holm(pvalues):
    # Nonestimable planned contrasts stay in the family with p=1.
    p = np.asarray([1 if x is None else x for x in pvalues], float)
    order = np.argsort(p)
    corrected = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out = np.empty(len(p));out[order] = corrected
    return out.tolist()
