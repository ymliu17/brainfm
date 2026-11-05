import sys,unittest
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from roi_signal_qc import assess,require_usable
from model_input_protocol import build_input
from baselines.fc_features import compute_fc

class SparseRoiTests(unittest.TestCase):
 def data(self,rois=424,n=175):
  return torch.randn(rois,n,generator=torch.Generator().manual_seed(42))+10
 def test_allow_sparse_but_reject_extensive_for_both_atlases(self):
  for n in [424,450]:
   x=self.data(n);limit=int(.05*n);x[:limit]=3
   self.assertEqual(require_usable(x,160)['n_dead'],limit)
   x[limit]=3
   with self.assertRaisesRegex(ValueError,'extensive_constant_rois'):require_usable(x,160)
 def test_nonfinite_is_never_replaced_with_signal(self):
  x=self.data();x[0,4]=float('nan')
  with self.assertRaisesRegex(ValueError,'nonfinite'):build_input(x,qc_policy='sparse')
 def test_padding_is_excluded_and_no_imputation(self):
  x=self.data();x[:6]=3;y=build_input(x,200,qc_policy='sparse')
  self.assertTrue(torch.equal(y[:6],torch.zeros_like(y[:6])))
  self.assertTrue(torch.equal(y[:,175:],torch.zeros_like(y[:,175:])))
  torch.testing.assert_close(y[6:,:175],x[6:]-x[6:].mean(1,keepdim=True))
  self.assertEqual(assess(x,200)['n_dead'],6);self.assertFalse(assess(x,200)['strict_pass'])
 def test_crop_can_reveal_unusable_window(self):
  x=self.data();x[:22,:160]=3
  self.assertEqual(assess(x,160)['n_dead_full'],0)
  with self.assertRaisesRegex(ValueError,'extensive_constant_rois'):build_input(x,160,qc_policy='sparse')
 def test_fc_guard_retains_valid_edges(self):
  x=self.data(20);x[0]=3;y=build_input(x,175,qc_policy='sparse')
  with np.errstate(divide='ignore',invalid='ignore'):fc=compute_fc(y.numpy()[None])[0]
  self.assertTrue(np.isfinite(fc).all());self.assertTrue((fc[0]==0).all())
  np.testing.assert_allclose(fc[1:,1:],np.corrcoef(y[1:].numpy()),atol=1e-7)

if __name__=='__main__':unittest.main()
