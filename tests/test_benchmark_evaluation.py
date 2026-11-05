import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import benchmark_evaluation as ev
from thirdorder import TuckerInput, transfer_features


class BenchmarkRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.groups = np.repeat(np.arange(30), 2)
        self.x = self.rng.normal(size=(60, 8))
        self.y = self.x[:, 0] + self.rng.normal(size=60)

    def test_nested_subject_isolation(self):
        prepared = ev.prepare_nested([self.x], [3], self.y, self.groups, 0)
        seen = []
        for tr, te, _, itr, iva, _ in prepared:
            self.assertFalse(set(self.groups[tr]) & set(self.groups[te]))
            self.assertFalse(set(self.groups[itr]) & set(self.groups[iva]))
            self.assertTrue(set(itr) | set(iva) <= set(tr))
            seen.extend(te)
        self.assertEqual(sorted(seen), list(range(60)))
        self.assertTrue(np.isfinite(ev.predict_prepared(prepared, self.y)).all())

    def test_heldout_values_cannot_change_training_transform(self):
        tr, te = np.arange(40), np.arange(40, 60)
        a, _ = ev.transform_blocks([self.x], [3], tr, te)
        changed = self.x.copy()
        changed[te] += 10000
        b, _ = ev.transform_blocks([changed], [3], tr, te)
        np.testing.assert_allclose(a, b)

    def test_inner_validation_does_not_fit_its_transform(self):
        prep = ev.prepare_nested([self.x], [3], self.y, self.groups, 0)
        tr, te, _, itr, iva, (inner_train, _) = prep[0]
        changed = self.x.copy()
        changed[iva] += 10000
        prep2 = ev.prepare_nested([changed], [3], self.y, self.groups, 0)
        np.testing.assert_allclose(inner_train, prep2[0][-1][0])

    def test_bootstrap_resamples_whole_subjects_and_reports_actual_se(self):
        pa = self.rng.normal(size=60)
        pb = self.y + self.rng.normal(size=60)
        obs, tail, ci, detail = ev.paired_bootstrap(
            self.y, pa, pb, 50, groups=self.groups, return_details=True)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(50):
            ix = np.concatenate([np.flatnonzero(self.groups == g)
                                 for g in rng.choice(np.arange(30), 30, replace=True)])
            diffs.append(ev.score(self.y[ix], pb[ix]) - ev.score(self.y[ix], pa[ix]))
        self.assertAlmostEqual(detail['bootstrap_se'], np.std(diffs, ddof=1))
        np.testing.assert_allclose(ci, np.percentile(diffs, [2.5, 97.5]))
        self.assertEqual(detail['n_subjects'], 30)
        self.assertGreater(tail, 0)

    def test_tucker_source_basis_and_nested_training_invariance(self):
        raw = self.rng.normal(size=(20, 2, 5, 24))
        valid = np.full((20, 2), 24)
        data = TuckerInput(raw, valid, 3, 'diff')
        a, b = data[:12].fit_transform_pair(data[12:])
        changed = TuckerInput(raw[12:] * 100 + self.rng.normal(size=raw[12:].shape), valid[12:], 3, 'diff')
        a2, _ = data[:12].fit_transform_pair(changed)
        np.testing.assert_allclose(a, a2)
        basis, (t1, t2) = transfer_features(raw[:12, 0], [raw[12:, 0], raw[12:, 0]], 3)
        np.testing.assert_allclose(t1, t2)
        self.assertEqual(basis.shape, (3, 5))

    def test_permutation_add_one_and_matching_statistic(self):
        from rct_probe import cv_score, permutation_p
        self.assertEqual(ev.permutation_tail(1., [0., .1, .2]), .25)
        with patch('rct_probe.GRID', (1.,)):
            obs = cv_score(self.x, self.y, False, n_pca=3)[0]
            p, _, _ = permutation_p(self.x, self.y, False, obs, 3, n_pca=3)
            self.assertGreaterEqual(p, .25)
            with self.assertRaises(ValueError):
                permutation_p(self.x, self.y, False, obs + .1, 3, n_pca=3)

    def test_tucker_probe_nested_path(self):
        from rct_probe import cv_score
        raw = self.rng.normal(size=(30, 2, 5, 24))
        data = TuckerInput(raw, np.full((30, 2), 24), 3, 't1')
        with patch('rct_probe.GRID', (1.,)):
            score, _, predictions = cv_score(data, self.y[:30], False, seeds=(0,), n_pca=2)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(len(predictions[0]), 30)

    def test_diagnosis_direction_preserves_raw_result(self):
        from confirm_matched_axis import FAM, block, two_sided
        rows = [dict(family=FAM, arm='CN/AD', cohort='trial', mem_level_rho=-.3, mem_level_n=80)]
        with contextlib.redirect_stdout(io.StringIO()):
            result = block('control', [('CN/AD', 'mem_level')], rows, False)[0]
        self.assertLess(result['raw_z'], 0)
        self.assertGreater(result['z'], 0)
        self.assertEqual(result['per_cohort']['trial'], -.3)
        self.assertEqual(result['p'], two_sided(result['raw_z']))



if __name__ == '__main__':
    unittest.main()
