import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from unified_supplements import absolute_null, projected_similarity
from readout_sensitivity import identification_scores

def test_bijection_null_matches_explicit_relabeling_with_ties():
    sim=np.array([[2.,2.,0.,-1.],[0.,1.,1.,0.],[0.,1.,3.,0.],[2.,0.,0.,2.]])
    n=211;seed=19;rng=np.random.default_rng(seed);observed=identification_scores(sim)
    scores=[identification_scores(sim[:,rng.permutation(4)]) for _ in range(n)]
    actual=absolute_null(sim,seed,n)
    for metric in ('top1','I_diff'):
        values=np.array([r[metric] for r in scores]);hits=int((values>=observed[metric]-1e-12).sum())
        assert actual[metric]['exceedances']==hits
        assert actual[metric]['p']==(1+hits)/(n+1)
        np.testing.assert_allclose(actual[metric]['null_mean'],values.mean(),atol=1e-14)

def test_constant_features_receive_fractional_chance_credit():
    out=absolute_null(np.ones((8,8)),1,100)
    assert out['top1']['score']==.125 and out['top1']['p']==1
    assert out['I_diff']['score']==0 and out['I_diff']['p']==1

def test_projection_is_independent_of_participant_order():
    rng=np.random.default_rng(9);a=rng.normal(size=(7,30));b=rng.normal(size=(7,30));p=np.array([3,0,1,2,5,6,4])
    x=projected_similarity(a,b);y=projected_similarity(a[p],b[p])
    np.testing.assert_allclose(y,x[p][:,p],atol=1e-7)
    np.testing.assert_allclose(projected_similarity(a,a).diagonal(),1,atol=1e-7)
