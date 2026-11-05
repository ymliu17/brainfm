import sys
from pathlib import Path
from dataclasses import dataclass
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from model_input_protocol import build_input, build_cohort


def test_retained_frames_only_and_no_gain_normalization():
    x = torch.tensor([[3.,5.,7.,100.,200.],[9.,13.,17.,-100.,-200.]])
    y = build_input(x, 3)
    torch.testing.assert_close(y, torch.tensor([[-2.,0.,2.],[-4.,0.,4.]]))
    torch.testing.assert_close(build_input(7*x+123, 3), 7*y)


def test_pad_after_centering_without_mutating_source():
    x = torch.tensor([[10.,12.,14.],[20.,24.,28.]])
    original = x.clone()
    y = build_input(x, 5)
    torch.testing.assert_close(y[:,:3], x-x.mean(1,keepdim=True))
    assert torch.equal(y[:,3:], torch.zeros(2,2))
    assert torch.equal(x, original)


def test_cohort_ignores_storage_padding_and_handles_visits_independently():
    @dataclass
    class Cohort:
        bold: torch.Tensor
        valid: torch.Tensor
    b = torch.full((1,2,5,2),float('nan'))
    b[0,:,:3,0] = torch.tensor([[1.,3.,5.],[10.,20.,30.]])
    b[0,:,:,1] = torch.tensor([[1.,2.,4.,100.,200.],[4.,8.,16.,200.,400.]])
    co = Cohort(b,torch.tensor([[3,5]]))
    new = build_cohort(co,4)
    torch.testing.assert_close(new.bold[0,:,:,0],build_input(b[0,:,:3,0],4))
    torch.testing.assert_close(new.bold[0,:,:,1],build_input(b[0,:,:,1],4))
    assert new.valid.tolist()==[[3,4]]
    assert torch.isnan(co.bold[0,:,3:,0]).all()
