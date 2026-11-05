import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'src')]
import numpy as np
import run_intervention_alignment as r


def data():
    rng=np.random.default_rng(84)
    return rng.normal(size=(24,7)),rng.normal(size=(24,7)),rng.normal(size=(70,7))


def test_coral_does_not_fit_heldout_participants():
    pre,post,ref=data();train=np.arange(18)
    first=r.aligned_inputs(pre,post,train,ref,'coral')
    changed_pre=pre.copy();changed_post=post.copy()
    changed_pre[23]+=900;changed_post[23]-=500
    second=r.aligned_inputs(changed_pre,changed_post,train,ref,'coral')
    for mode in r.ip.MODES:
        np.testing.assert_allclose(first[mode][:23],second[mode][:23],atol=1e-12)


def test_post_visits_do_not_estimate_calibration():
    pre,post,ref=data();train=np.arange(18)
    first=r.aligned_inputs(pre,post,train,ref,'coral')
    second=r.aligned_inputs(pre,post*10+100,train,ref,'coral')
    np.testing.assert_allclose(first['baseline'],second['baseline'],atol=1e-12)


def test_shared_affine_transform_preserves_linear_change():
    pre,post,ref=data();train=np.arange(18)
    alignment=r.harmonization.Alignment(ref).fit(pre[train],'coral')
    result=r.aligned_inputs(pre,post,train,ref,'coral')
    np.testing.assert_allclose(result['change'],(post-pre)@alignment.matrix,atol=1e-12)
    np.testing.assert_allclose(result['baseline_plus_change'],np.c_[result['baseline'],result['change']])
    none=r.aligned_inputs(pre,post,train,ref,'source_pca_unaligned')
    np.testing.assert_array_equal(none['baseline'],pre)
    np.testing.assert_array_equal(none['change'],post-pre)
