import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from paper_robustness import shortened,earliest_indices,covariate_pair,weighted_pair,plans


class RobustnessTests(unittest.TestCase):
    def test_short_window_recenters_real_frames_and_pads_afterward(self):
        x=torch.arange(2*4*200,dtype=torch.float32).reshape(2,4,200)
        x-=x.mean(-1,keepdim=True)
        y=shortened(x)
        self.assertEqual(y.shape,x.shape)
        torch.testing.assert_close(y[:,:,:80].mean(-1),torch.zeros(2,4))
        self.assertEqual(torch.count_nonzero(y[:,:,80:]).item(),0)
        shifted=shortened(x+77)
        torch.testing.assert_close(y,shifted)

    def test_earliest_scan_is_chronological_and_not_label_selected(self):
        m=pd.DataFrame(dict(subject=['a','b','a','a'],visit=['M120','M006','M060','BL'],scan=['a120','b6','a60','a0'],MEM=[1,2,3,np.nan]))
        self.assertEqual(earliest_indices(m).tolist(),[3,1])

    def test_covariates_use_training_scale_and_categories(self):
        m=pd.DataFrame(dict(age=[10.,20.,1000.],site=['A','B','unseen']))
        a,b=covariate_pair(m,['age','site'],np.array([0,1]),np.array([2]))
        np.testing.assert_allclose(a[:,0],[-1,1]);self.assertEqual(b[0,0],197)
        self.assertEqual(a.shape[1],3);np.testing.assert_allclose(b[0,1:],[-1,-1])

    def test_participant_weighting_invariant_to_duplicate_visit(self):
        y=np.array([1.,3.,2.,4.,6.,5.]);a=np.array([1.,2.,4.,3.,5.,6.]);b=y+.1
        first=weighted_pair(y,a,b,np.arange(6),n_boot=256)
        ix=np.r_[np.arange(6),0,0,0]
        duplicated=weighted_pair(y[ix],a[ix],b[ix],np.arange(6)[ix],n_boot=256)
        self.assertAlmostEqual(first['reference_score'],spearmanr(y,a).statistic)
        self.assertAlmostEqual(first['delta'],duplicated['delta'])
        np.testing.assert_allclose(first['ci95'],duplicated['ci95'],atol=1e-12)

    def test_complete_case_only_applies_to_covariate_panel(self):
        n=30;m=pd.DataFrame(dict(subject=[str(i) for i in range(n)],scan=[str(i) for i in range(n)],visit=['M000']*n,
            eligible=True,age=70.,sex=1.,native_frames=200,mean_fd=.1,MEM=np.arange(n),EF=np.arange(n)))
        m.loc[0,'mean_fd']=np.nan
        p={x['key']:x for x in plans('adni',m)}
        self.assertEqual(len(p['confounds_MEM_baseline']['indices']),29)
        self.assertEqual(len(p['duration_MEM_baseline']['indices']),30)


if __name__=='__main__':unittest.main()
