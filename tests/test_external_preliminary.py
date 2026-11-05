import sys, unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_external_preliminary import normalize, axis_training_target

class ExternalInputTests(unittest.TestCase):
 def test_missing_stage_is_excluded_not_imputed(self):
  y=np.array([1.,3.,10.,14.,8.,100.]);stages=['CN','CN','MCI','MCI','Dementia',np.nan]
  mask,target=axis_training_target(y,stages,'all_residualized')
  np.testing.assert_array_equal(mask,[True,True,True,True,True,False])
  np.testing.assert_allclose(target[:5],[-1,1,-2,2,0])
  np.testing.assert_array_equal(y,[1,3,10,14,8,100])
  mask,_=axis_training_target(y,stages,'MCI')
  np.testing.assert_array_equal(mask,[False,False,True,True,False,False])
 def test_padding_is_zero_and_gain_preserved(self):
  torch.manual_seed(1)
  x=torch.randn(424,175)*torch.linspace(1,3,424)[:,None]
  result=normalize(x)
  self.assertEqual(tuple(result.shape),(424,200))
  self.assertTrue(torch.equal(result[:,175:],torch.zeros(424,25)))
  self.assertTrue(torch.allclose(result[:,:175].mean(1),torch.zeros(424),atol=1e-6))
  expected=x.std(1,unbiased=False)/x.std(1,unbiased=False).median()
  self.assertTrue(torch.allclose(result[:,:175].std(1,unbiased=False),expected,atol=1e-6))
 def test_constant_roi_rejected(self):
  x=torch.randn(424,200);x[0]=0
  with self.assertRaisesRegex(ValueError,'constant_roi'):normalize(x)
 def test_nonfinite_rejected(self):
  x=torch.randn(424,200);x[0,0]=float('nan')
  with self.assertRaisesRegex(ValueError,'invalid_tensor'):normalize(x)
 def test_crop_does_not_reorder_rois(self):
  torch.manual_seed(2);x=torch.randn(424,300)
  expected=((x-x.mean(1,keepdim=True))/x.std(1,unbiased=False).median())[:,:200]
  self.assertTrue(torch.equal(normalize(x),expected))
if __name__=='__main__':unittest.main()
