import sys
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_readout_task_extension as ext
from benchmark_evaluation import grouped_splits
from sklearn.model_selection import GroupShuffleSplit


def template():
    g = np.repeat(np.arange(30).astype(str), 2)
    y = np.repeat(np.arange(30) % 2, 2).astype(float)
    m = pd.DataFrame(dict(scan=[str(i) for i in range(60)], subject=g, DX=y))
    folds = []
    for s in (0, 1, 2):
        for f, (tr, te) in enumerate(grouped_splits(y, g, s)):
            a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=s).split(tr, groups=g[tr]))
            folds.append(dict(seed=s, fold=f, train=tr.tolist(), test=te.tolist(),
                              inner_train=tr[a].tolist(), validation=tr[b].tolist()))
    return dict(status='complete', covariates=[], scan_ids=m.scan.tolist(), subjects=g.tolist(),
                y=y.tolist(), split_audit=folds), m


def test_parent_samples_labels_and_grouped_splits_are_enforced():
    old, m = template()
    ix, y, g = ext.audit_template(old, m, 'DX')
    np.testing.assert_array_equal(ix, np.arange(60))
    changed = m.copy()
    changed.loc[0, 'DX'] = 1
    with pytest.raises(AssertionError):
        ext.audit_template(old, changed, 'DX')
    bad = copy.deepcopy(old)
    bad['split_audit'][0]['inner_train'].append(bad['split_audit'][0]['validation'][0])
    with pytest.raises(ValueError, match='inner partition'):
        ext.audit_template(bad, m, 'DX')


def test_cpu_streamed_pca_preserves_binary_probe_probabilities():
    from paper_completion import project_block
    rng = np.random.default_rng(72)
    x = rng.normal(size=(38, 5100))
    tr, te = np.arange(28), np.arange(28, 38)
    y = np.arange(28) % 2
    a, b = ext.frozen.project_indices(x, tr, te, 8, 'cpu')
    c, d = project_block(x[tr], x[te], 8)
    first = LogisticRegression(C=.1, max_iter=5000, tol=1e-10).fit(a, y).predict_proba(b)
    second = LogisticRegression(C=.1, max_iter=5000, tol=1e-10).fit(c, y).predict_proba(d)
    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)


def test_scope_and_correction_families_do_not_select_models_or_cohorts():
    jobs = ext.jobs()
    assert len(jobs) == 63
    assert len({j['id'] for j in jobs}) == 63
    assert {j['task'] for j in jobs if j['cohort']=='adni'} == {'AGE', 'SEX', 'DX'}
    assert ext.family('DX', 'native') != ext.family('AD_DX', 'native')
    assert ext.family('DX', 'native') != ext.family('DX', 'sf100_fc')
    for model in ext.frozen.MODELS:
        assert 'layer3' not in ext.frozen.readout_names(model)
