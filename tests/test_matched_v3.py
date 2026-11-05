import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from fm_matched_v3.models import patch_coverage, weighted_mean
from fm_matched_v3.probes import TuckerBlock, transform
from fm_matched_v3.data import hrv_ef_scores, endpoints
from prepare_hrv_memory import recall_score
from fm_matched_v3.evaluate import clinical_features


class CommonReadoutTests(unittest.TestCase):
    def test_hrv_clinical_model_does_not_require_nonexistent_baseline_memory(self):
        m=pd.DataFrame(dict(baseline_EF=[90.,100.],baseline_MEM=[np.nan,np.nan],
                            age_category=['61-63','64-66'],sex=['male','female'],arm=['increase','decrease']))
        values=clinical_features(m,'hrv_pre')
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[:,0],[90.,100.])

    def test_delayed_recall_counts_unique_correct_images_and_preserves_zero(self):
        table=pd.DataFrame(dict(participant_id=[7001]*4,correct=[1,1,0,2],
                                final_code=['pic1','pic1','NO','NO'],set=['A']*4))
        result=recall_score(table,'sub-7001')
        self.assertEqual(result['MEM_delayed_recall'],1)
        self.assertEqual(result['duplicate_correct_responses'],1)
        self.assertEqual(recall_score(table.iloc[2:],'sub-7001')['MEM_delayed_recall'],0)
        with self.assertRaises(ValueError):recall_score(table.iloc[:0],'sub-7001')
        with self.assertRaises(ValueError):recall_score(table,'sub-7002')

    def test_hrv_composite_is_complete_case_standard_score_mean(self):
        table = pd.DataFrame({'pre_flanker_standardScore':[90,90],
                              'pre_LSWM_standardScore':[100,np.nan],
                              'pre_PCPS_standardScore':[110,110],
                              'pre_flanker_compScore':[7,8]})
        values = hrv_ef_scores(table, 'pre')
        self.assertEqual(values.iloc[0],100)
        self.assertTrue(np.isnan(values.iloc[1]))
        self.assertEqual(endpoints('hrv_pre'),['EF'])

    def test_coverage_preserves_input_area_and_excludes_padding(self):
        for size, patch in [(432, 16), (434, 14)]:
            weights = patch_coverage(424, 200, size, patch)
            self.assertEqual(weights.sum(), 424*200)
            self.assertTrue((weights == 0).any())
            self.assertEqual(len(weights), (size//patch)**2)

    def test_padding_tokens_do_not_change_pooled_features(self):
        x = torch.tensor([[[1., 2.], [3., 4.], [1e8, -1e8]]], requires_grad=True)
        y = weighted_mean(x, [1, 3, 0])
        torch.testing.assert_close(y, torch.tensor([[2.5, 3.5]]))
        y.sum().backward()
        torch.testing.assert_close(x.grad[0, 2], torch.zeros(2))
        self.assertGreater(float(x.grad[0, 0].sum()), 0)

    def test_invalid_weights_fail(self):
        for w in ([0, 0], [1], [1, -1]):
            with self.assertRaises(ValueError):
                weighted_mean(torch.ones(1, 2, 3), w)

    def test_tucker_training_transform_excludes_test_scans(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(12, 5, 20))
        original = TuckerBlock(x, rank=3)
        modified = x.copy(); modified[8:] *= 100
        modified[8:] += rng.normal(size=(4, 5, 20))*300
        a, _ = transform([original], [2], np.arange(8), np.arange(8,12))
        b, _ = transform([TuckerBlock(modified, rank=3)], [2], np.arange(8), np.arange(8,12))
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_tucker_basis_receives_training_only(self):
        x = np.random.default_rng(1).normal(size=(12,5,20))
        import fm_matched_v3.probes as probes
        ref = probes.shared_coskew_basis(x[:8], rank=3)
        actual = TuckerBlock(x, 3).basis(np.arange(8))
        np.testing.assert_allclose(actual.T@actual, ref.T@ref, atol=1e-10)

    def test_tucker_subset_does_not_change_index_mapping(self):
        x = np.random.default_rng(5).normal(size=(15,5,20))
        full = TuckerBlock(x, 3)
        sub = full[np.array([1,3,5,7,9,11])]
        a = sub.basis(np.array([0,2,4]))
        b = full.basis(np.array([1,5,9]))
        np.testing.assert_allclose(a,b,atol=1e-10)
        self.assertIs(sub.cache,full.cache)


if __name__ == '__main__':
    unittest.main()
