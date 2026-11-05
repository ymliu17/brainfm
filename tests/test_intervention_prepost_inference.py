import sys
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from intervention_prepost_inference import bootstrap_metrics, primary_batch, swaps
from intervention_prepost import metrics


class InferenceTests(unittest.TestCase):
    def test_bootstrap_multiclass_metrics_equal_explicit_repeated_participants(self):
        rng=np.random.default_rng(718);y=np.tile(np.arange(3),10)
        p=rng.dirichlet([1,2,3],len(y));w=rng.multinomial(len(y),np.ones(len(y))/len(y),size=50)
        batched=bootstrap_metrics(y,p,w,['a','b','c'])
        for i,weights in enumerate(w):
            ix=np.repeat(np.arange(len(y)),weights);expected=metrics(y[ix],p[ix],['a','b','c'])
            for name,value in expected.items():self.assertAlmostEqual(batched[name][i],value,places=12)

    def test_bootstrap_regression_reranks_ties_and_repeats(self):
        y=np.array([1,1,3,4,5,6.]);p=np.array([2,2,5,3,4,7.])
        w=np.array([[2,0,1,1,2,0],[0,1,1,2,0,2]])
        batched=bootstrap_metrics(y,p,w,[])
        for i,weights in enumerate(w):
            ix=np.repeat(np.arange(len(y)),weights)
            for name,value in metrics(y[ix],p[ix],[]).items():self.assertAlmostEqual(batched[name][i],value,places=12)

    def test_swaps_move_complete_multiclass_vectors(self):
        rng=np.random.default_rng(982);y=np.tile(np.arange(3),4)
        a=rng.dirichlet([1,1,1],len(y));b=rng.dirichlet([1,1,1],len(y));classes=['a','b','c']
        seed=6;count=127
        observed=metrics(y,b,classes)['auc']-metrics(y,a,classes)['auc']
        masks=np.random.default_rng(seed).random((count,len(y)))<.5;exceed=0
        for mask in masks:
            pa=np.where(mask[:,None],b,a);pb=np.where(mask[:,None],a,b)
            np.testing.assert_allclose(pa.sum(1),1);np.testing.assert_allclose(pb.sum(1),1)
            delta=metrics(y,pb,classes)['auc']-metrics(y,pa,classes)['auc']
            exceed+=abs(delta)>=abs(observed)-1e-12
        result=swaps(y,a,b,classes,seed,n_perm=count)
        self.assertEqual(result['permutation_exceedances'],exceed)
        self.assertEqual(result['permutation_p'],(1+exceed)/(1+count))

    def test_binary_batches_and_exact_identity(self):
        y=np.array([0,1,0,1,0,1]);p=np.array([[.8,.2],[.1,.9],[.7,.3],[.6,.4],[.2,.8],[.3,.7]])
        self.assertAlmostEqual(primary_batch(y,p,['0','1'])[0],metrics(y,p,['0','1'])['auc'])
        r=swaps(y,p,p,['0','1'],42,n_perm=20)
        self.assertTrue(r['exact_identity']);self.assertEqual(r['permutation_p'],1)


if __name__=='__main__':unittest.main()
