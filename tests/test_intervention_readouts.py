import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import pytest
import run_intervention_readouts as r


def test_paired_outcome_or_participant_changes_rejected():
    plan={'tasks':{'EF_change':{'subjects':['a','b'],'y':[.2,.4],'classes':[]}}}
    row=dict(task='EF_change',subjects=['a','b'],y=[.2,.4],classes=[])
    r.validate_row(row,plan)
    with pytest.raises(ValueError,match='subjects'):
        r.validate_row(dict(row,subjects=['b','a']),plan)
    with pytest.raises(ValueError,match='y'):
        r.validate_row(dict(row,y=[.2,.5]),plan)


def test_training_pca_not_affected_by_other_test_rows():
    rng=np.random.default_rng(10);x=rng.normal(size=(24,35))
    train=np.arange(18);test=np.arange(18,24)
    a,b=r.ip.project(x,train,test)
    changed=x.copy();changed[23]+=10000
    aa,bb=r.ip.project(changed,train,test)
    np.testing.assert_allclose(a,aa,atol=1e-12)
    np.testing.assert_allclose(b[:-1],bb[:-1],atol=1e-12)


def test_multiclass_swap_identity_and_symmetry():
    y=np.tile(np.arange(3),6);rng=np.random.default_rng(21)
    a=rng.dirichlet(np.ones(3),len(y));b=rng.dirichlet(np.ones(3),len(y))
    same=r.stats.swaps(y,a,a,['a','b','c'],9,n_perm=100)
    assert same['permutation_p']==1 and same['exact_identity']
    ab=r.stats.swaps(y,a,b,['a','b','c'],9,n_perm=100)
    ba=r.stats.swaps(y,b,a,['a','b','c'],9,n_perm=100)
    assert ab['permutation_p']==ba['permutation_p']
