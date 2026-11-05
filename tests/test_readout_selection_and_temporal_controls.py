import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_nested_readout_selection as nested
import run_temporal_order_controls as temporal


@pytest.mark.parametrize('model',temporal.base.MODELS)
def test_shuffle_preserves_fc_amplitudes_padding_and_metadata(model):
    gen=torch.Generator().manual_seed(12)
    rois=424 if model.startswith('brainlm') else 450
    total=457 if model=='semantoks' else 402 if model=='brainharmonix_f' else rois
    x=torch.zeros(total,200)
    signal_rois=400 if model=='brainharmonix_f' else rois
    x[:signal_rois,:37]=torch.randn(signal_rois,37,generator=gen)
    x[0]=0  # Dead ROI: invariance check must not generate NaNs.
    if model=='brainharmonix_f':x[400,0]=37;x[401,0]=10;x[401,1]=3.528
    meta=SimpleNamespace(native_frames=37,resampled_frames=37)
    a,check=temporal.shuffle_row(x,model,meta,211,'subject_visit',True)
    b,_=temporal.shuffle_row(x,model,meta,211,'subject_visit',True)
    c,_=temporal.shuffle_row(x,model,meta,212,'subject_visit',True)
    assert torch.equal(a,b) and not torch.equal(a,c)
    assert torch.equal(a[:,37:],x[:,37:])
    assert torch.equal(a[signal_rois:],x[signal_rois:])
    assert check['fc_max_abs_error']<1e-10


def test_common_time_permutation_is_not_independent_roi_shuffle():
    x=torch.arange(40).float()[None].repeat(424,1)+torch.arange(424)[:,None]*100
    a,_=temporal.shuffle_row(x,'brainlm13',SimpleNamespace(native_frames=40),211,'scan')
    np.testing.assert_array_equal((a-torch.arange(424)[:,None]*100).numpy(),np.repeat(a[0:1].numpy(),424,axis=0))


def test_readout_tie_prefers_primary_and_finite_validation():
    rows=[dict(readout='native',validation_rho=.2),dict(readout='connectivity',validation_rho=.2),
          dict(readout='spatial_norm',validation_rho=None)]
    assert nested.choose_readout(rows)['readout']=='native'
    rows[1]['validation_rho']=.3
    assert nested.choose_readout(rows)['readout']=='connectivity'


def test_grid_uses_only_supplied_training_and_validation_targets():
    rng=np.random.default_rng(21)
    ai=rng.normal(size=(20,3));bi=rng.normal(size=(8,3));ytrain=ai[:,0];yval=bi[:,0]
    best,grid=nested.choose_grid(ai,bi,ytrain,yval)
    assert len(grid)==len(nested.GRID)
    assert best['ridge_alpha']==1/best['c']
    # This API cannot receive outer-test outcomes; future observations do not
    # participate in either transform fitting or this selection function.
    again,_=nested.choose_grid(ai.copy(),bi.copy(),ytrain.copy(),yval.copy())
    assert best==again
