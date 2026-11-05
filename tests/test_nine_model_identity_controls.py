import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import run_nine_model_identity_controls as c


def test_real_window_summaries_ignore_padding():
    x = np.random.default_rng(4).normal(size=(12, 30)).astype('float32')
    padded = np.pad(x, ((0, 0), (0, 170)))
    for a, b in zip(c.old.input_summaries(x, 30), c.old.input_summaries(padded, 30)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not (ROOT/"runs/brainlm13_identity_controls_v1/pairs/adni.json").exists(), reason="Requires private cached galleries")
def test_pair_membership_matches_every_existing_model():
    for co in c.COHORTS:
        pair = c.pair(co)
        assert len(pair['subjects']) == len(set(pair['subjects']))
        for model in c.MODELS:
            prior = c.read(c.frozen.RUN/'identity'/model/f'{co}.json')
            assert prior['subjects'] == pair['subjects']
            for mode in c.MODES:
                for readout in ('native', 'connectivity'):
                    assert len(prior['scores'][mode][readout]['per_subject_top1']) == len(pair['subjects'])


def test_random_reference_is_average_scores_not_selected_seed():
    # Equal averaging includes every registered seed, even when one performs best.
    pretrained_gain = np.array([.3, .2, .4])
    random_gains = np.array([[.1, -.1, .2], [.5, .6, .8], [0, 0, 0], [.1, .2, .3], [.1, .2, .3]])
    expected = pretrained_gain-random_gains.mean(0)
    d = c.rs.paired_identification(random_gains.mean(0), pretrained_gain, seed=17, n_boot=99, n_perm=99)
    np.testing.assert_allclose(d['delta'], expected.mean())


def test_same_final_layer_connectivity_definition():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 5, 8)).astype('float32')
    edges = c.frozen.connectivity(x)
    expected = np.stack([np.corrcoef(row)[np.triu_indices(5, 1)] for row in x])
    np.testing.assert_allclose(edges, expected, atol=1e-6)
