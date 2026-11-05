"""Nested probes supporting genuinely training-fitted FC-Tucker coordinates."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from benchmark_evaluation import grouped_splits, predict_prepared
from thirdorder import shared_coskew_basis, tucker_fc_feature


class TuckerBlock:
    def __init__(self, scans, rank=64, indices=None, cache=None):
        self.scans = np.asarray(scans)
        self.indices = np.arange(len(scans)) if indices is None else np.asarray(indices)
        self.cache = {} if cache is None else cache
        self.rank = min(rank, self.scans.shape[1])
        self.shape = (len(self.indices), self.rank*(self.rank-1)//2)

    def __getitem__(self, ix):
        return TuckerBlock(self.scans, self.rank, self.indices[ix], self.cache)

    def basis(self, train):
        # Individual-scan Gram matrices contain no fitted cross-subject statistics.
        # Cache them once, then aggregate strictly the requested training rows.
        if 'grams' not in self.cache:
            n,p,_ = self.scans.shape
            grams = np.empty((n,p,p), dtype=np.float64)
            for i,x in enumerate(self.scans):
                z = np.asarray(x,dtype=np.float64)
                z = z-z.mean(1,keepdims=True)
                z = z/np.maximum(z.std(1,keepdims=True),1e-8)
                g = z.T@z
                grams[i] = (z@(g*g)@z.T)/(z.shape[1]**2)
                if i%200==0:print('FC-Tucker per-scan moments',i,n,flush=True)
            self.cache['grams'] = grams
        # This sum has exactly the same definition as shared_coskew_basis.
        acc = self.cache['grams'][self.indices[train]].mean(0)
        _,vectors = np.linalg.eigh((acc+acc.T)*.5)
        return vectors[:,::-1][:,:self.rank].T

    def fit_pair(self, train, test):
        # No evaluation scan contributes to the shared co-skewness basis.
        basis = self.basis(train)
        return (tucker_fc_feature(self.scans[self.indices[train]], basis=basis),
                tucker_fc_feature(self.scans[self.indices[test]], basis=basis))


def transform(blocks, dims, train, test):
    aa, bb = [], []
    for block, k in zip(blocks, dims):
        a, b = block.fit_pair(train, test) if isinstance(block, TuckerBlock) else (block[train], block[test])
        sc = StandardScaler().fit(a)
        a, b = sc.transform(a), sc.transform(b)
        if k < a.shape[1]:
            pc = PCA(n_components=min(k, len(train)-1), random_state=0).fit(a)
            a, b = pc.transform(a), pc.transform(b)
        aa.append(a); bb.append(b)
    return np.hstack(aa), np.hstack(bb)


def oof_predictions(blocks, dims, y, seed, groups):
    groups = np.asarray(groups)
    prepared = []
    for tr, te in grouped_splits(y, groups, seed):
        a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(tr, groups=groups[tr]))
        it, iv = tr[a], tr[b]
        prepared.append((tr, te, transform(blocks, dims, tr, te), it, iv, transform(blocks, dims, it, iv)))
    return predict_prepared(prepared, y)


def oof(block, y, groups, k=100):
    return np.mean([oof_predictions([block], [k], y, seed, groups) for seed in (0, 1, 2)], axis=0)
