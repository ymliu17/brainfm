import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import sample_alignment as sa
import readout_sensitivity as rs


class PopulationTests(unittest.TestCase):
    def setUp(self):
        self.m=pd.DataFrame(dict(scan=['a1','a2','b1','b2','c1','c2'],
            subject=['a','a','b','b','c','c'],visit=['T1','T2']*3,eligible=True))
        # a lacks follow-up imaging; b lacks baseline imaging.
        self.available=np.array([True,False,False,True,True,True])

    def test_missing_followup_does_not_exclude_baseline_prediction(self):
        with patch.object(sa,'metadata',return_value=self.m),patch.object(sa,'available',return_value=self.available):
            np.testing.assert_array_equal(sa.common_mask(self.m,'psopt'),[True,False,False,False,True,False])

    def test_identification_requires_both_scans_and_rebuilds_gallery(self):
        with patch.object(rs,'metadata',return_value=self.m),patch.object(sa,'available',return_value=self.available):
            subjects,(_,a),(_,b)=sa.paired_rows('psopt')
        self.assertEqual(subjects,['c']);self.assertEqual(a.tolist(),[4]);self.assertEqual(b.tolist(),[5])
        # Removing a gallery member changes a retained query's identification.
        sim=np.array([[1.,0.,0.],[0.,.4,.8],[0.,.3,.9]])
        old=rs.identification_scores(sim)['per_subject_top1']
        new=rs.identification_scores(sim[np.ix_([0,1],[0,1])])['per_subject_top1']
        self.assertFalse(np.array_equal(old[[0,1]],new))

    def test_reordered_metadata_rejected(self):
        with patch.object(sa,'metadata',return_value=self.m),patch.object(sa,'available',return_value=self.available):
            with self.assertRaises(ValueError):sa.common_mask(self.m.iloc[::-1],'cogte')

    def test_refit_uses_new_training_population_and_grouped_folds(self):
        from benchmark_evaluation import prepare_nested,predict_prepared
        rng=np.random.default_rng(81)
        groups=np.repeat(np.arange(24),2);x=rng.normal(size=(48,6));y=x[:,0]+rng.normal(size=48)
        keep=groups!=5
        def fit(xx,yy,gg):
            prep=prepare_nested([xx],[4],yy,gg,0)
            for tr,te,_,it,iv,_ in prep:
                self.assertFalse(set(gg[tr])&set(gg[te]))
                self.assertFalse(set(gg[it])&set(gg[iv]))
            return predict_prepared(prep,yy)
        original=fit(x,y,groups)
        refitted=fit(x[keep],y[keep],groups[keep])
        self.assertTrue(np.isfinite(refitted).all())
        self.assertFalse(np.allclose(original[keep],refitted))

    def test_ambiguous_inference_cache_does_not_overwrite_other_predictor(self):
        ident=dict(cohort='hrv_pre',task='recall',family='test',reference='frozen',candidate='adapted',metric='rho')
        c1=dict(ident,y=[1,2,3],a=[1,2,3],b=[3,2,1],subjects=['a','b','c'])
        c2=dict(c1,b=[1,2,3])
        old=[dict(ident,statistics={'delta':-2}),dict(ident,statistics={'delta':0})]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'parent/inference/test.json'
            sa.save(p,dict(rows=old))
            with patch.object(sa,'PARENT_PAPER',root/'parent'),patch.object(sa,'PAPER',root/'new'),patch.object(sa.pc,'paired',side_effect=[{'delta':-2},{'delta':0}]) as paired:
                sa.cached_infer('test',[c1,c2])
                self.assertEqual(paired.call_count,2)
                rows=sa.read(root/'new/inference/test.json')['rows']
                self.assertEqual([r['statistics']['delta'] for r in rows],[-2,0])

if __name__=='__main__':unittest.main()
