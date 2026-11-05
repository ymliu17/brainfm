import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from paper_paired_stats import weighted_scores,score_batch,paired,holm
from paper_completion import align,average,project_block


class PairedTests(unittest.TestCase):
    def test_weighted_ranks_match_explicit_cluster_bootstrap_with_ties(self):
        y=np.array([1.,1.,4.,2.,5.,5.]);p=np.array([.1,.4,.4,.3,.2,.2])
        weights=np.array([[2,2,0,1,3,3],[0,0,3,2,1,1]])
        actual=weighted_scores(y,p,weights)
        for i,w in enumerate(weights):
            ix=np.repeat(np.arange(len(y)),w)
            self.assertAlmostEqual(actual[i],spearmanr(y[ix],p[ix]).statistic,places=12)
        binary=np.array([0.,0.,1.,1.,0.,1.])
        actual=weighted_scores(binary,p,weights,True)
        for i,w in enumerate(weights):
            ix=np.repeat(np.arange(len(y)),w)
            self.assertAlmostEqual(actual[i],roc_auc_score(binary[ix],p[ix]),places=12)

    def test_batched_scores_match_scalar_scores(self):
        y=np.array([0.,1.,0.,1.,0.,1.]);p=np.array([[1.,2.,2.,4.,5.,6.],[6.,2.,5.,1.,4.,4.]])
        np.testing.assert_allclose(score_batch(y,p),[spearmanr(y,z).statistic for z in p])
        np.testing.assert_allclose(score_batch(y,p,True),[roc_auc_score(y,z) for z in p])

    def test_identical_methods_have_no_difference(self):
        y=np.arange(30,dtype=float);p=y+.1*np.sin(y)
        r=paired(y,p,p,np.repeat(np.arange(15),2),n_boot=128,n_perm=128)
        self.assertEqual(r['delta'],0);self.assertEqual(r['permutation_p'],1)
        self.assertEqual(r['ci95'],[0.,0.])

    def test_alignment_requires_scan_identity(self):
        a={'subjects':['a','a'],'scan_ids':['v1','v2'],'y':[1,1],'prediction':[0,1]}
        with self.assertRaises(ValueError):align(a,dict(a,scan_ids=['v2','v1']))
        with self.assertRaises(ValueError):average([a,dict(a,y=[1,2])])

    def test_projection_does_not_fit_held_out_data(self):
        rng=np.random.default_rng(11);a=rng.normal(size=(25,12));b=rng.normal(size=(7,12))
        first,_=project_block(a,b,5);second,_=project_block(a,b*100+1000,5)
        np.testing.assert_allclose(first,second,rtol=0,atol=0)

    def test_holm_retains_nonestimable_contrasts(self):
        np.testing.assert_allclose(holm([.01,.03,None]),[.03,.06,1])


if __name__=='__main__':unittest.main()
