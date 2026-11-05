import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('posttraining_pca',ROOT/'scripts/run_posttraining_pca_sensitivity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import numpy as np


class PCASensitivityTests(unittest.TestCase):
    def test_cached_projection_matches_original_and_ignores_test_distribution(self):
        x=np.random.default_rng(4).normal(size=(40,70));tr=np.arange(25);te=np.arange(25,40)
        project=m.cached_projector()
        for k in (5,10,20):
            with patch.object(m.ip,'K',k):expected=m.ip.project(x,tr,te)
            actual=project(x,tr,te,k)
            for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=0,atol=0)
        shifted=x.copy();shifted[te]+=100
        np.testing.assert_allclose(project(x,tr,te,10)[0],project(shifted,tr,te,10)[0],rtol=0,atol=0)

    def test_ensemble_requires_three_aligned_seeds(self):
        row=dict(task='EF_change',mode='change',setting='imaging_only',subjects=['a','b'],y=[1,2],classes=[],
                 predictions={'fixed5':[.1,.2],'inner_selected':[.2,.4]})
        with self.assertRaises(ValueError):m.ensemble([row,row])
        bad={**row,'subjects':['b','a']}
        with self.assertRaises(ValueError):m.ensemble([row,row,bad])
        actual=m.ensemble([row,row,row])
        np.testing.assert_allclose(actual['predictions']['inner_selected'],[.2,.4])

    @unittest.skipUnless(m.frozen_path('brainlm13','psopt').exists(), 'Requires private experiment artifacts; not distributed')
    def test_cached_frozen_reproduction(self):
        # Exercise the full regression, multiclass-capable classifier, clinical
        # transforms, all nested folds and PC choices on an actual parent panel.
        model='brainlm13';co='psopt'
        expected=m.row_index(m.read(m.frozen_path(model,co)))
        with tempfile.TemporaryDirectory() as d:
            with patch.object(m.r,'RUN',Path(d)),patch.object(m.r,'project_k',m.cached_projector()):
                m.r.pca(model,co)
            result=m.read(Path(d)/'pca'/model/(co+'.json'))
        m.validate_rows(result['rows'],co)
        for key,row in m.row_index(result).items():
            for name in ('fixed5','inner_selected'):
                np.testing.assert_allclose(row['predictions'][name],expected[key]['predictions'][name],rtol=1e-7,atol=1e-8)
            self.assertEqual(row['selected'],expected[key]['selected'])


if __name__=='__main__':unittest.main()
