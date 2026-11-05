"""Third-order (co-skewness) features: FC computed in a kappa_3-optimal basis.

WHY
---
Reconstruction-pretrained fMRI foundation models optimise MSE, which allocates capacity to
whatever carries the most variance -- cardiac and respiratory cycles, head motion, scanner
drift.

  Marraffini, Mahuas, Borrell, Shevchenko & Wassermann, "The Variance Brain Foundation
  Models Forgot: Third-Order Statistics Predict Cognition Where Billion-Parameter Models
  Fail", arXiv 2606.04010.

They measure what survives pretraining: the second cumulant is partially preserved
(Log-Cholesky relative error 29-47% on reconstruction, though above 1.0 on HCP) while the
THIRD cumulant is destroyed (NMSE 2.6-6.2x the input signal energy), and cognition-relevant
information concentrates there. They report BrainLM-650M r=0.003, Brain-JEPA r=0.004, raw
FC r=0.393, and their FC-Tucker at r=0.571 over 200 folds -- no pretraining, CPU only.

NAMING. `FC-Tucker` is THEIR name for the feature and `tucker` is the label used for it
throughout this project; it is their method, not ours. Two differences from their setup
travel with every number here: this implementation uses rank 64 where their best is
R* = 333, and it is applied to ADNI CN/AD and the RCT outcomes rather than to HCP
cognition.

That is directly relevant here twice over. As a BASELINE it is the strongest published
competitor and reviewers will find it. As a SUPERVISION TARGET on ADNI it is a mechanism-
driven post-training task: it asks the encoder to retain exactly the statistic the
pretraining objective provably discards.

THE CHEAP IDENTITY THAT MAKES THIS TRACTABLE
--------------------------------------------
The co-skewness tensor S[i,j,k] = (1/T) sum_t z_it z_jt z_kt has P^3 entries -- 76 million
for a 424-ROI atlas, 305 MB per scan in float32. Never form it. Only the mode-1 subspace is
needed, and its Gram matrix has a closed form in the TIME domain:

    (S_(1) S_(1)^T)[i,i'] = (1/T^2) sum_{t,s} z_it z_i's (z_t . z_s)^2
                          = (1/T^2) [ Z (G o G) Z^T ][i,i']      with G = Z^T Z, o elementwise

G is T x T (80 or 140 here), so the whole thing costs O(P T^2 + P^2 T) and runs on CPU.
The leading eigenvectors of that P x P matrix span the same subspace HOSVD would return.
"""
from __future__ import annotations

import numpy as np


def coskew_basis(z: np.ndarray, rank: int) -> np.ndarray:
    """[rank, P] mode-1 subspace of the co-skewness tensor of a [P, T] recording.

    Returned as rows so it applies as `U @ z`. Uses the time-domain Gram identity above;
    the P^3 tensor is never materialised.
    """
    T = z.shape[1]
    G = z.T @ z                      # [T, T]
    M = (z @ (G * G) @ z.T) / (T * T)  # [P, P], = S_(1) S_(1)^T
    M = 0.5 * (M + M.T)             # symmetrise against round-off
    w, V = np.linalg.eigh(M)
    return V[:, ::-1][:, :rank].T    # descending eigenvalue order


def shared_coskew_basis(x3: np.ndarray, valid: np.ndarray | None = None,
                        rank: int = 64, standardize: bool = True) -> np.ndarray:
    """[rank, P] co-skewness basis pooled over scans.

    THE BASIS MUST BE SHARED. Fitting it per scan makes the resulting edges meaningless
    across subjects: component 1 of one subject's basis is not the same brain direction as
    component 1 of another's, and eigenvector sign and ordering are arbitrary per scan, so a
    linear model has nothing stable to weight. Measured on ADNI CN/AD (n=498, 424 ROIs,
    rank 64): a per-scan basis scores 0.5062 -- chance -- against 0.6680 for this pooled
    basis and 0.6949 for plain FC. The group-level convention also matches how the
    comparable published features are built (principal gradients, group PCA).

    For inductive evaluation, pass training rows only. Fitting on held-out scans
    uses their distribution and is transductive, even without labels.
    """
    P = x3.shape[1]
    acc = np.zeros((P, P), dtype=np.float64)
    for i in range(len(x3)):
        t = int(valid[i]) if valid is not None else x3.shape[2]
        z = _prep(x3[i, :, :t], standardize)
        G = z.T @ z
        acc += (z @ (G * G) @ z.T) / (z.shape[1] ** 2)
    acc /= len(x3)
    acc = 0.5 * (acc + acc.T)
    w, V = np.linalg.eigh(acc)
    return V[:, ::-1][:, :rank].T


def _prep(a, standardize):
    z = np.asarray(a, dtype=np.float64)
    z = z - z.mean(axis=1, keepdims=True)
    if standardize:
        z = z / np.maximum(z.std(axis=1, keepdims=True), 1e-8)
    return z


def tucker_fc_feature(x3: np.ndarray, valid: np.ndarray | None = None,
                      rank: int = 64, standardize: bool = True,
                      basis: np.ndarray | None = None) -> np.ndarray:
    """[N, rank*(rank-1)/2] Fisher-z FC edges computed in each scan's co-skewness basis.

    `standardize` z-scores each ROI over time before estimating the third cumulant, which
    makes it a skewness rather than a raw third moment. It is the default because it makes
    this feature scale-invariant in the same way FC is, which is the fair comparison: an
    un-standardised basis would let between-subject amplitude enter the FEATURE here while
    the FC baseline it is compared against cannot see amplitude at all.

    `basis` is the [rank, P] shared basis from `shared_coskew_basis`. When omitted it is
    fitted on `x3` itself, which is the right default for a single cohort and keeps callers
    from silently getting the per-scan version that scores at chance -- see that function.
    """
    if basis is None:
        basis = shared_coskew_basis(x3, valid, rank, standardize)
    rank = basis.shape[0]
    n = len(x3)
    iu = np.triu_indices(rank, k=1)
    out = np.empty((n, len(iu[0])), dtype=np.float32)
    for i in range(n):
        t = int(valid[i]) if valid is not None else x3.shape[2]
        z = _prep(x3[i, :, :t], standardize)
        y = basis @ z                                # [rank, T]
        y = y - y.mean(axis=1, keepdims=True)
        sd = np.maximum(y.std(axis=1, keepdims=True), 1e-12)
        c = (y / sd) @ (y / sd).T / y.shape[1]
        # Same Fisher-z and same clipping as transfer_baseline.fc_feature, so the two
        # features differ only in the basis they are computed in.
        out[i] = np.arctanh(np.clip(c[iu], -0.999999, 0.999999)).astype(np.float32)
    return out


def transfer_features(training_scans, evaluation_scans, rank=64):
    """Fit one source-only basis and apply it to every evaluation block.

    Call inside each outer fold; never include evaluation scans in basis fitting.
    """
    basis = shared_coskew_basis(training_scans, rank=rank)
    return basis, [tucker_fc_feature(x, basis=basis) for x in evaluation_scans]


class TuckerInput:
    """Deferred scan features: each nested fold fits its own training-only basis."""
    def __init__(self, scans, valid, rank=64, mode="t1"):
        self.scans, self.valid = np.asarray(scans), np.asarray(valid)
        self.rank, self.mode = rank, mode
        width = min(rank, self.scans.shape[2])
        self.shape = (len(scans), width * (width - 1) // 2 * (2 if mode == "t1t2" else 1))

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        return TuckerInput(self.scans[index], self.valid[index], self.rank, self.mode)

    def fit_transform_pair(self, other):
        basis = shared_coskew_basis(self.scans[:, 0], self.valid[:, 0], self.rank)
        def encode(data):
            fs = [tucker_fc_feature(data.scans[:, t], data.valid[:, t], basis=basis)
                  for t in (0, 1)]
            return (fs[0] if self.mode == "t1" else fs[1] if self.mode == "t2" else
                    np.hstack(fs) if self.mode == "t1t2" else fs[1] - fs[0])
        return encode(self), encode(other)
