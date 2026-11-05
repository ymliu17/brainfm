"""Scientific reporting invariants for the six-model multiplicity update."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import joint_inference as u


def row(model='fmrilm', cohort='act', delta=.1, p=.001, **kwargs):
    r = dict(model=model, cohort=cohort, source='adni', objective='supervised',
             adaptation_target='MEM', task='MEM_baseline', analysis_method='FM',
             reference='frozen', candidate='adapted', mode='none', setting='none',
             scope='none', axis_source='none', contrast='none', protocol='primary',
             evaluation_scope='independent_of_posttraining_source', metric='rho',
             family='adaptation_MEM', reference_score=.1, candidate_score=.1+delta,
             delta=delta, permutation_p=p, holm_p=p, n=50)
    r.update(kwargs)
    return r


def normalized(records):
    rows = []
    for i, r in enumerate(records):
        u.add(rows, r, 'cognitive_adaptation', u.ROOT / 'source.json', i)
    return rows


def test_all_models_share_one_family():
    f = u.adjust(normalized([row(model=m, p=.01) for m in u.MODELS]))
    assert f.family.nunique() == 1
    assert f.family_size_six_models.eq(6).all()
    np.testing.assert_allclose(f.holm_p_six_models, .06)
    assert not f.significant_six_models.any()


def test_finite_permutation_resolution_and_missing_tests():
    p, rule = u.finite_p(dict(permutation_exceedances=0, permutations_valid=20000, permutation_p=0))
    assert p == 1/20001
    assert rule == 'finite_count_reconstruction'
    assert u.finite_p(dict(permutation_p=None))[0] == 1


def test_supervised_targets_map_to_same_original_family():
    a = row(target='MEM')
    b = row(target='EF')
    assert u.adaptation_family(a) == u.adaptation_family(b)


def test_source_exposure_is_not_external_replication():
    exposed = row(source='adni', cohort='adni', objective='native_ssl')
    external = row(source='nacc', cohort='adni', objective='native_ssl')
    assert u.adaptation_family(exposed) != u.adaptation_family(external)
    f = u.adjust(normalized([row(cohort='act', evaluation_scope='source_exposed_descriptive'),
                             row(cohort='breathe', evaluation_scope='source_exposed_descriptive')]))
    assert not u.consistency(f).label.str.contains('external_gain').any()


def test_duplicate_identity_cannot_hide_different_scores():
    same = normalized([row(), row()])
    keep, discarded = u.deduplicate(same)
    assert len(keep) == len(discarded) == 1
    different = normalized([row(), row(delta=.2)])
    with pytest.raises(ValueError, match='unequal candidate_score'):
        u.deduplicate(different)


def test_no_replicated_benefit_from_nonsignificance_or_multiple_visits():
    f = u.adjust(normalized([row(cohort='hrv_pre', p=.001), row(cohort='hrv_post', p=.001)]))
    c = u.consistency(f)
    assert c.n_cohorts.iloc[0] == 1
    assert c.label.iloc[0] == 'single_cohort_only'
    f = u.adjust(normalized([row(cohort='act', p=.9), row(cohort='breathe', p=.8)]))
    c = u.consistency(f)
    assert c.corrected_gain_cohorts.iloc[0] == 0
    assert c.label.iloc[0] == 'directionally_consistent_positive'


def test_replication_requires_same_strategy_and_no_corrected_loss():
    f = u.adjust(normalized([row(cohort='act'), row(cohort='breathe')]))
    assert u.consistency(f).label.iloc[0] == 'consistent_external_gain'
    f = u.adjust(normalized([row(cohort='act'), row(cohort='breathe'), row(cohort='psopt', delta=-.1)]))
    assert u.consistency(f).label.iloc[0] == 'mixed_or_unresolved'
    f = u.adjust(normalized([row(cohort='act'), row(cohort='breathe', adaptation_target='EF')]))
    assert u.consistency(f).label.eq('single_cohort_only').all()
