import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from sklearn.preprocessing import StandardScaler
from embedding_harmonization import Alignment,subject_mean

def test_moment_matching_and_local_standardization_invariance():
 r=np.random.default_rng(3);s=r.normal(size=(60,5))*3+4;t=r.normal(size=(40,5));test=r.normal(size=(10,5))
 a=Alignment(s).fit(t,'moments');aligned=a.transform(t)
 np.testing.assert_allclose(aligned.mean(0),s.mean(0),atol=1e-12)
 np.testing.assert_allclose(aligned.std(0),s.std(0),atol=1e-12)
 np.testing.assert_allclose(StandardScaler().fit(t).transform(test),StandardScaler().fit(aligned).transform(a.transform(test)),atol=1e-12)

def test_coral_identity_and_no_test_fit():
 r=np.random.default_rng(4);s=r.normal(size=(80,5));t=r.normal(size=(30,5))+3
 a=Alignment(s).fit(s,'coral');np.testing.assert_allclose(a.transform(s),s,atol=1e-12)
 a=Alignment(s).fit(t,'coral');test=r.normal(size=(10,5));baseline=a.transform(test)
 a.transform(test*100+30)
 np.testing.assert_array_equal(a.transform(test),baseline)

def test_source_moments_weight_subjects_equally():
 z=np.array([[0.,2.],[2.,4.],[10.,20.]])
 np.testing.assert_array_equal(subject_mean(z,['a','a','b']),[[1,3],[10,20]])
