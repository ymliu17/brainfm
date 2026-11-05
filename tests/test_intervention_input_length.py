import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts')]
import numpy as np
import pytest
import torch
import run_intervention_input_length as r


def test_short_context_ignores_later_frames_and_recenters():
    torch.manual_seed(31)
    x=torch.randn(2,5,200);changed=x.clone();changed[:,:,80:]+=1000
    a=r.robustness.shortened(x);b=r.robustness.shortened(changed)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(a[:,:,:80].mean(-1),torch.zeros(2,5),atol=1e-7,rtol=0)
    assert torch.count_nonzero(a[:,:,80:])==0


def test_replay_detects_changed_or_misaligned_features():
    rng=np.random.default_rng(7);x=rng.normal(size=(4,50))
    assert r.reproduction(x,x)['relative_rmse']==0
    with pytest.raises(ValueError,match='differs'):
        r.reproduction(x[::-1],x)
    with pytest.raises(ValueError,match='differs'):
        r.reproduction(x*1.1,x)


def test_required_visits_union_task_specific_pairs():
    p={'tasks':{'MEM_change':{'pre_indices':[0,2],'post_indices':[1,3]},
                'arm':{'pre_indices':[2,4],'post_indices':[3,5]}}}
    assert r.required(p,'pre')==[0,2,4]
    assert r.required(p,'post')==[1,3,5]
