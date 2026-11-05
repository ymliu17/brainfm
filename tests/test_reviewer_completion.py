import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import reviewer_completion_tasks as r

def test_pca_selection_does_not_use_outer_test_labels():
    rng=np.random.default_rng(41);x=rng.normal(size=(60,25));y=x[:,0]+rng.normal(size=60)
    tr=np.arange(45);te=np.arange(45,60);it=np.arange(34);iv=np.arange(34,45)
    p,k=r.selected_pca_prediction(x,y,tr,te,it,iv,[])
    altered=y.copy();altered[te]=rng.normal(100,50,len(te))
    q,j=r.selected_pca_prediction(x,altered,tr,te,it,iv,[])
    np.testing.assert_array_equal(p,q);assert k==j

def test_covariate_imputation_uses_training_median_and_unknown_site_is_safe():
    import pandas as pd
    m=pd.DataFrame({'age':[60.,70.,np.nan,999.],'site':['a','a','b','new']})
    a,b=r.covariates(m,['age','site'],np.array([0,1,2]),np.array([3]))
    altered=m.copy();altered.loc[3,'age']=-999
    c,_=r.covariates(altered,['age','site'],np.array([0,1,2]),np.array([3]))
    np.testing.assert_array_equal(a,c);assert np.isfinite(b).all();assert abs(a[2,0])<1e-12

def test_absolute_null_user_deferral_is_enforced(tmp_path,monkeypatch):
    import benchmark_completion_tasks as c
    (tmp_path/'execution_policy.json').write_text('{"absolute_nulls_enabled": false}')
    monkeypatch.setattr(c,'RUN',tmp_path)
    import pytest
    with pytest.raises(RuntimeError,match='deferred'):c.null_test('nacc','brainlm13','DX')

def test_same_window_fc_excludes_nonobserved_padding():
    from transfer_baseline import fc_feature
    rng=np.random.default_rng(4);x=rng.normal(size=(2,6,20))
    padded=np.pad(x,((0,0),(0,0),(0,10)))
    np.testing.assert_allclose(fc_feature(x,None),fc_feature(padded,np.array([20,20])),rtol=0,atol=0)
