import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_evidence_nulls as e

def test_exact_projection_matches_full_svd_predictions():
    rng=np.random.default_rng(44)
    for n,p in [(30,70),(90,25)]:
        x=rng.normal(size=(n,p));y=rng.normal(size=n);tr=np.arange(n-8);te=np.arange(n-8,n)
        a,b=e.projection(x,tr,te,5)
        s=StandardScaler().fit(x[tr]);pc=PCA(5,svd_solver='full').fit(s.transform(x[tr]))
        aa,bb=pc.transform(s.transform(x[tr])),pc.transform(s.transform(x[te]))
        np.testing.assert_allclose(Ridge(alpha=10).fit(a,y[tr]).predict(b),Ridge(alpha=10).fit(aa,y[tr]).predict(bb),atol=1e-11)

def test_whole_participant_trajectories_are_preserved():
    g=np.repeat(['a','b','c','d','e'],[2,2,3,3,1]);y=np.array([10,11,20,21,30,31,32,40,41,42,50])
    p=e.permuted(y,g,np.random.default_rng(9))
    for person in np.unique(g):
        z=p[g==person]
        assert any(np.array_equal(z,y[g==other]) for other in np.unique(g) if sum(g==person)==sum(g==other))
    assert sorted(p)==sorted(y)

def test_observed_and_null_refit_grouped_stratified_partitions():
    rng=np.random.default_rng(8);x=rng.normal(size=(60,12));y=rng.normal(size=60);g=np.repeat(np.arange(30),2)
    j=dict(task='MEM',kind='local',metric='rho',components=5)
    _,a=e.predictions(x,y,g,j);_,b=e.predictions(x,e.permuted(y,g,rng),g,j)
    assert any(not np.array_equal(u['test'],v['test']) for u,v in zip(a,b))
    for s in a+b:
        assert not set(g[s['train']]) & set(g[s['test']])
        assert not set(g[s['inner_train']]) & set(g[s['validation']])

def test_vectorized_ridge_refits_each_null_independently():
    rng=np.random.default_rng(9);a=rng.normal(size=(40,7));b=rng.normal(size=(8,7));y=rng.normal(size=(40,4))
    p=e.ridge_many(a,y,b,.1)
    for k in range(4):np.testing.assert_allclose(p[:,k],Ridge(alpha=10).fit(a,y[:,k]).predict(b),atol=1e-12)

def test_1999_chunks_cover_budget_once_including_short_final_chunk():
    for kind,cohort in [('local','adni'),('paired','act'),('physiology','hrv')]:
        intervals=e.chunk_ranges(dict(kind=kind,cohort=cohort,permutations=1999))
        assert intervals[0][0]==0 and intervals[-1][1]==1999
        assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
        assert sum(b-a for a,b in intervals)==1999

def test_scheduler_reserves_fast_slots_without_idling_after_a_lane_ends():
    from collections import deque
    fast=deque([('physiology',),('intervention',)]);heavy=deque([('adni',),('nacc',)])
    assert e.take_next(fast,heavy,0,88,16)==(('physiology',),'fast')
    assert e.take_next(fast,heavy,16,88,16)==(('adni',),'heavy')
    assert e.take_next(fast,deque(),16,88,16)==(('intervention',),'fast')
    assert e.take_next(deque(),heavy,0,88,16)==(('nacc',),'heavy')
