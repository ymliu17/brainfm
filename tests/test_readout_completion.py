import sys,copy
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_readout_completion as m


def test_reuse_rejects_different_samples_labels_or_splits():
    a=dict(scan_ids=['s1','s2'],subjects=['p1','p2'],y=[0.,1.],split_audit=[dict(train=[0],test=[1])])
    m.assert_aligned(a,copy.deepcopy(a),True)
    for key,value in [('scan_ids',['s2','s1']),('subjects',['p2','p1']),('y',[1.,0.]),('split_audit',[])]:
        b=copy.deepcopy(a);b[key]=value
        with pytest.raises((ValueError,AssertionError)):m.assert_aligned(a,b,True)


def test_transfer_reproduces_existing_source_axis_recipe():
    from fm_matrix.evaluate import fitted_axis
    rng=np.random.default_rng(17)
    x=rng.normal(size=(125,145)).astype(np.float32);y=rng.normal(size=125);t=rng.normal(size=(14,145)).astype(np.float32)
    pred=m.fitted_transfer(x,y,t)
    np.testing.assert_allclose(pred,fitted_axis(x,y,t),rtol=1e-10,atol=1e-10)
    # Other target participants must not affect a held-out participant's transform.
    changed=t.copy();changed[1:]+=200
    np.testing.assert_allclose(pred[0],m.fitted_transfer(x,y,changed)[0],rtol=1e-7,atol=1e-8)


def test_registered_scope_is_bounded_and_covers_every_model():
    jobs=m.jobs();assert len(jobs)==198 and len({j['id'] for j in jobs})==198
    assert sum(j['action']=='local' for j in jobs)==99
    assert sum(j['action']=='identity' for j in jobs)==63
    assert sum(j['action']=='transfer' for j in jobs)==36
    for model in m.frozen.MODELS:
        x=[j for j in jobs if j['model']==model and j['action']=='transfer']
        assert {(j['cohort'],j['task']) for j in x}=={(co,ep) for co in ('adni','nacc') for ep in ('MEM','EF')}
    assert m.atlas('brainharmonix_f')=='sch400'
    assert m.atlas('brainlm650')=='a424'
