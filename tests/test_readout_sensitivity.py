from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from readout_sensitivity import roi_readouts,identification_scores,similarity,paired_identification


class ReadoutTests(unittest.TestCase):
    def test_roi_embeddings_preserve_regional_axis(self):
        roi=torch.arange(450,dtype=torch.float32)[None,:,None,None]
        time=torch.arange(10,dtype=torch.float32)[None,None,:,None]
        x=(roi+time).expand(1,450,10,768)
        r,m,n=roi_readouts(x)
        self.assertEqual(tuple(r.shape),(1,450,768))
        torch.testing.assert_close(r[0,:,0],torch.arange(450,dtype=torch.float32)+4.5)
        torch.testing.assert_close(m,x.mean((1,2)))
        torch.testing.assert_close(n,(torch.arange(450,dtype=torch.float32)[None]+4.5)*768**.5)

    def test_identification_ties_are_not_id_order_dependent(self):
        d=identification_scores(np.ones((7,7)))
        self.assertAlmostEqual(d['top1'],1/7)
        np.testing.assert_allclose(d['per_subject_top1'],1/7)
        self.assertEqual(d['I_diff'],0)

    def test_identification_perfect_and_paired_zero(self):
        d=identification_scores(np.eye(8));self.assertEqual(d['top1'],1)
        p=paired_identification(d['per_subject_top1'],d['per_subject_top1'],n_boot=128,n_perm=128)
        self.assertEqual(p['permutation_p'],1);np.testing.assert_equal(p['ci95'],[0,0])

    def test_similarity_is_invariant_to_positive_scale_and_offset(self):
        rng=np.random.default_rng(1);a=rng.normal(size=(6,10));b=rng.normal(size=(6,10))
        np.testing.assert_allclose(similarity(a,b),similarity(a*3+5,b*2-1),atol=1e-12)


if __name__=='__main__':unittest.main()
