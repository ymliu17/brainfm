import importlib.util
from pathlib import Path
import unittest
import numpy as np
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('connectivity_readouts',ROOT/'scripts/run_connectivity_readouts.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ConnectivityReadoutTests(unittest.TestCase):
    def test_connectivity_is_feature_correlation_and_handles_constants(self):
        x=np.random.default_rng(5).normal(size=(2,7,19)).astype(np.float32)
        expected=np.stack([np.corrcoef(a)[np.triu_indices(7,1)] for a in x])
        np.testing.assert_allclose(m.connectivity(x),expected,rtol=2e-5,atol=2e-7)
        x[:,2]=3
        self.assertTrue(np.isfinite(m.connectivity(x)).all())

    def test_spatial_pooling_ignores_pure_padding_and_uses_boundary_overlap(self):
        # 8x8 image, 2x2 patches, central4x2 observed input.
        x=np.full((1,4,4,3),999.,dtype=np.float32)
        x[:,1:3,1,:]=2.;x[:,1:3,2,:]=4.
        actual=m.spatial_patch_mean(x.reshape(1,16,3),8,2,rois=4,frames=2)
        np.testing.assert_allclose(actual,np.full((1,2,3),3.))

    def test_streamed_pca_matches_full_svd_predictions_and_excludes_test_fit(self):
        from paper_completion import project_block
        rng=np.random.default_rng(42);x=rng.normal(size=(31,5100));tr=np.arange(23);te=np.arange(23,31)
        x[:,0]=1.;y=rng.normal(size=len(tr))
        a,b=m.project_indices(x,tr,te,k=7)
        c,d=project_block(x[tr],x[te],7)
        np.testing.assert_allclose(Ridge(alpha=10).fit(a,y).predict(b),Ridge(alpha=10).fit(c,y).predict(d),rtol=1e-7,atol=1e-8)
        altered=x.copy();altered[te]+=123
        aa,_=m.project_indices(altered,tr,te,k=7)
        np.testing.assert_allclose(a@a.T,aa@aa.T,rtol=1e-12,atol=1e-10)

    def test_layer_and_stage_controls_are_not_in_final_readout_family(self):
        for model in m.MODELS:
            self.assertNotIn('image_encoder_mean',m.readout_names(model))
            for ref,cand,family in m.contrasts(model):
                if family.startswith('final_readouts'):
                    self.assertIn(cand,m.readout_names(model))
                    self.assertNotIn(cand,('layer3','layer4_raw'))


if __name__=='__main__':unittest.main()
