"""Scientific invariants for the explicitly separate window-reencoding panel."""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from new_fm.physiology import window_input
from run_new_fm_physiology import planned

@pytest.mark.parametrize('model',['semantoks','fmrilm'])
def test_no_neighboring_windows_enter_encoder_input(model):
    torch.manual_seed(3);x=torch.randn(450,160);order=np.arange(450)
    a,n=window_input(x,2,2.4,model,order)
    changed=x.clone();changed[:,:40]*=100;changed[:,60:]-=100
    b,m=window_input(changed,2,2.4,model,order)
    assert n==m==24
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.count_nonzero(a[:,24:])==0

@pytest.mark.parametrize('tr,n',[(1.01,11),(1.375,14),(2.4,24)])
def test_window_duration_is_resampled_not_stretched(tr,n):
    x=torch.randn(450,160)
    z,actual=window_input(x,7,tr,'fmrilm',np.arange(450))
    assert actual==n and z.shape==(450,180)
    assert torch.count_nonzero(z[:,n:])==0
    with pytest.raises(ValueError):window_input(x[:,:159],7,tr,'fmrilm',np.arange(450))

def test_queue_reuses_checkpoints_and_stages_frozen_before_adapted():
    jobs=planned();lookup={j['id']:j for j in jobs}
    assert len(jobs)==154 and len(lookup)==154
    assert len({j['output'] for j in jobs})==154
    assert set(j['action'] for j in jobs)=={'extract','evaluate','report'}
    for j in jobs:
        if j['action']=='extract':
            assert 'new_fm_v1/features/' in j['external_output']
            assert j['dependencies']==([] if j['phase']=='frozen' else ['report_frozen'])
    done=set()
    while len(done)<len(jobs):
        ready={j['id'] for j in jobs if set(j['dependencies'])<=done}-done
        assert ready
        done|=ready
