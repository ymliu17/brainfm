import sys
from pathlib import Path
import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from intervention_prepost import project, clinical_transform, fit_predict, metrics


class PrePostTests(unittest.TestCase):
    def test_dual_pca_matches_exact_svd_and_ignores_test_distribution(self):
        x=np.random.default_rng(817).normal(size=(40,200))
        tr=np.arange(30);te=np.arange(30,40)
        a,b=project(x,tr,te)
        sc=StandardScaler().fit(x[tr])
        pc=PCA(n_components=5,svd_solver='full').fit(sc.transform(x[tr]))
        aa=pc.transform(sc.transform(x[tr]));bb=pc.transform(sc.transform(x[te]))
        np.testing.assert_allclose(a@a.T,aa@aa.T,atol=1e-8)
        np.testing.assert_allclose(a@b.T,aa@bb.T,atol=1e-8)
        x[te]*=10000
        changed,_=project(x,tr,te)
        np.testing.assert_allclose(a,changed,atol=1e-10)

    def test_constant_features_have_a_finite_null_projection(self):
        a,b=project(np.ones((12,100)),np.arange(8),np.arange(8,12))
        self.assertFalse(a.any());self.assertFalse(b.any())
        self.assertTrue(np.isfinite(a).all())

    def test_clinical_imputation_scaling_and_categories_are_train_only(self):
        p=dict(clinical=dict(age=[60,70,None,999],arm=['A','B','A','unseen']),
               numeric_covariates=['age'],categorical_covariates=['arm'])
        tr=np.arange(3);te=np.array([3])
        a,b=clinical_transform(p,tr,te)
        self.assertTrue(np.isfinite(a).all())
        p['clinical']['age'][3]=-9999;p['clinical']['arm'][3]='another unseen'
        changed,_=clinical_transform(p,tr,te)
        np.testing.assert_array_equal(a,changed)

    def test_multiclass_probability_columns_and_auc(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(90,5));y=np.tile(np.arange(3),30)
        p=fit_predict(x[:60],y[:60],x[60:],1,['A','B','C'])
        self.assertEqual(p.shape,(30,3));np.testing.assert_allclose(p.sum(1),1)
        m=metrics(y[60:],p,['A','B','C'])
        self.assertTrue(0<=m['auc']<=1)
        self.assertTrue(np.isfinite(list(m.values())).all())


if __name__=='__main__':unittest.main()
