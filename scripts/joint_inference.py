#!/usr/bin/env python3
"""Audit and jointly correct all available six-model benchmark contrasts.

This is a retrospective multiplicity/reporting update, not a new experiment or
prospective registration. Source artifacts and fitted predictions are immutable.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paper_paired_stats import holm

OLD = ROOT / 'runs/paper_results_2026-10-14/complete_comparisons'
NEW = ROOT / 'runs/new_fm_v1/reports/all'
RUN = ROOT / 'runs/six_model_inference_v1'
OUT = ROOT / 'results/generated/six_model_evaluation'
MODELS = ('brainlm13', 'brainlm111', 'brainlm650', 'brainjepa', 'semantoks', 'fmrilm')
ADDED = MODELS[-2:]
INPUTS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    if path.suffix == '.csv':
        return pd.read_csv(path).replace({np.nan: None}).to_dict('records')
    return json.loads(path.read_text())


def clean(v, default='none'):
    return default if v is None or (isinstance(v, float) and np.isnan(v)) else v


def finite_p(r):
    """Use finite-sample P when exceedance/valid counts were archived."""
    b, n = r.get('permutation_exceedances'), r.get('permutations_valid')
    if b is not None and n is not None and np.isfinite(b) and np.isfinite(n) and n > 0:
        return (float(b) + 1) / (float(n) + 1), 'finite_count_reconstruction'
    p = r.get('permutation_p')
    if p is None or not np.isfinite(p):
        return 1., 'nonestimable_or_unavailable_kept_in_family'
    if not 0 <= p <= 1:
        raise ValueError('Invalid P value')
    return float(p), 'archived_raw_p'


def bounds(r):
    ci = r.get('ci95')
    if isinstance(ci, str):
        ci = ast.literal_eval(ci)
    if ci is not None:
        return ci
    return r.get('ci_low'), r.get('ci_high')


def outcome(task):
    if str(task).startswith('MEM'):
        return 'MEM'
    if str(task).startswith('EF'):
        return 'EF'
    return task


def canonical_task(task, co, method='FM'):
    if task in ('MEM', 'EF'):
        return task + ('_concurrent_post' if co == 'hrv_post' else '_baseline')
    if task == 'MEM_delayed_recall':
        return ('MEM_delayed_recall_post_scan_association' if co == 'hrv_post'
                else 'MEM_delayed_recall_baseline_prediction')
    return task


def adaptation_family(r):
    co, task, obj = r['cohort'], r['task'], r['objective']
    if co.startswith('hrv') and (task.startswith('MEM') or co == 'hrv_post'):
        setting = 'hrv_exploratory'
    elif r.get('section') == 'axes' or '_axis_to_' in task:
        setting = 'direct_transfer'
    elif co in ('adni', 'nacc'):
        setting = 'observational'
    elif 'followup' in task:
        setting = 'followup'
    else:
        setting = 'intervention_baseline'
    family = 'adaptation_' + obj + '_' + setting + '_' + outcome(task)
    if obj == 'native_ssl' and co == r.get('source'):
        family += '_source_exposed_descriptive'
    return family


def clinical_family(task, objective):
    setting = 'hrv_exploratory' if 'delayed_recall' in task else 'followup'
    base = 'clinical_increment_' + setting + '_' + outcome(task)
    return base if objective == 'frozen' else base + '_' + objective


def clinical_test(job):
    """Small CPU-only conditional test; safe to distribute across local workers."""
    from paper_paired_stats import paired
    info, y, a, b, subjects = job
    seed = int(hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:8], 16)
    return dict(info, **paired(y, a, b, subjects, seed=seed))


def original_adapted_clinical():
    """Complete the same clinical-increment question for the original four FMs."""
    from concurrent.futures import ProcessPoolExecutor
    base = ROOT / 'runs/sample_alignment_v1/paper_completion_v1/canonical'
    jobs, hashes = [], {}
    for model in MODELS[:4]:
        for source in ('adni', 'nacc'):
            for obj, target in [('supervised', 'MEM'), ('supervised', 'EF'), ('native_ssl', 'none')]:
                records = []
                for seed in (42, 43, 44):
                    path = base / f'{model}_{source}_{obj}_{target}_s{seed}.json'
                    records.append(read(path)['rows'])
                    hashes[str(path.relative_to(ROOT))] = digest(path)
                maps = [{(r['cohort'], r['task'], r['method']): r for r in rs} for rs in records]
                for k, r in maps[0].items():
                    if r['method'] != 'clinical+FM':
                        continue
                    rs = [m[k] for m in maps]
                    cs = [m[(k[0], k[1], 'clinical')] for m in maps]
                    for q in rs + cs:
                        for field in ('subjects', 'scan_ids', 'y'):
                            if q[field] != r[field]:
                                raise ValueError('Unpaired original clinical ensemble: ' + field)
                    info = dict(model=model, source=source, objective=obj,
                                adaptation_target=target, cohort=r['cohort'], task=r['task'],
                                reference='clinical', candidate='clinical+FM',
                                analysis_method='clinical+FM', family=clinical_family(r['task'], obj))
                    jobs.append((info, r['y'], np.mean([q['prediction'] for q in cs], axis=0),
                                 np.mean([q['prediction'] for q in rs], axis=0), r['subjects']))
    cache = RUN / 'original_adapted_clinical_inference.json'
    if cache.exists():
        data = json.loads(cache.read_text())
        if data['inputs_sha256'] != hashes:
            raise ValueError('Original clinical inference inputs changed')
    else:
        print(f'Completing {len(jobs)} original-model clinical-increment tests on 4 CPU workers; no training.', flush=True)
        with ProcessPoolExecutor(max_workers=4) as pool:
            stats = list(pool.map(clinical_test, jobs))
        data = dict(inputs_sha256=hashes, rows=stats, n_boot=5000, n_perm=20000,
                    status='complete', retrospective=True, trained_models=False)
        cache.write_text(json.dumps(data, indent=2) + '\n')
    return cache, read(cache)['rows']


def add(rows, r, panel, source_path, index, **updates):
    a = dict(r)
    a.update(updates)
    a['panel'] = panel
    a['input_file_provenance'] = str(Path(source_path).relative_to(ROOT))
    a['input_row'] = index
    a['archived_family'] = r.get('family')
    a['archived_holm_p'] = r.get('holm_p')
    a['archived_raw_p'] = r.get('permutation_p')
    a['raw_p'], a['raw_p_rule'] = finite_p(r)
    a['cohort'] = clean(a.get('cohort'))
    a['model'] = clean(a.get('model', a.get('candidate')))
    a['source'] = clean(a.get('source'))
    a['objective'] = clean(a.get('objective'), 'frozen')
    a['adaptation_target'] = clean(a.get('adaptation_target', a.get('target')))
    a['evaluation_scope'] = clean(a.get('evaluation_scope'), 'independent_external')
    a['analysis_method'] = clean(a.get('analysis_method'), 'FM')
    a['protocol'] = clean(a.get('protocol'), 'primary_scan_level')
    a['candidate'] = clean(a.get('candidate'), 'adapted' if a['objective'] != 'frozen' else a['model'])
    a['reference'] = clean(a.get('reference'), 'frozen' if a['objective'] != 'frozen' else 'sf100_fc')
    a['task'] = clean(a.get('task', a.get('endpoint', a.get('domain'))))
    a['mode'] = clean(a.get('mode'))
    a['setting'] = clean(a.get('setting'))
    a['scope'] = clean(a.get('scope'))
    a['axis_source'] = clean(a.get('axis_source'))
    a['contrast'] = clean(a.get('contrast'))
    a['ci_low'], a['ci_high'] = bounds(r)
    a['family'] = panel + '::' + a['family']
    # Stable identity excludes random resampling seed and previous correction.
    fields = ['panel', 'model', 'cohort', 'source', 'objective', 'adaptation_target',
              'task', 'analysis_method', 'reference', 'candidate', 'mode', 'setting',
              'scope', 'axis_source', 'contrast', 'protocol', 'metric', 'family']
    a['hypothesis_id'] = hashlib.sha256(json.dumps([a.get(k) for k in fields], sort_keys=True).encode()).hexdigest()[:24]
    rows.append(a)


def build_rows():
    rows = []
    # Already expanded frozen local families contain original frozen controls.
    path = NEW / 'local_comparisons.json'
    for i, r in enumerate(read(path)):
        model = clean(r.get('model', r.get('candidate')))
        if model not in ADDED:
            if r['reference'] == 'clinical':
                add(rows, r, 'local_clinical_increment', path, i, model=r.get('candidate'),
                    family=clinical_family(r['task'], 'frozen'),
                    objective='frozen', protocol='sf100_exact_probe')
            else:
                add(rows, r, 'local_frozen', path, i, model=r.get('candidate'), objective='frozen')
            continue
        obj = r['objective']
        if obj == 'frozen':
            if r['reference'] == 'clinical':
                add(rows, r, 'local_clinical_increment', path, i,
                    family=clinical_family(r['task'], 'frozen'),
                    analysis_method='clinical+FM', protocol='sf100_exact_probe')
            else:
                add(rows, r, 'local_frozen', path, i)
        elif r['reference'] == 'clinical':
            add(rows, r, 'local_clinical_increment', path, i,
                family=clinical_family(r['task'], obj),
                analysis_method='clinical+FM', protocol='sf100_exact_probe')
        else:
            co, task = r['cohort'], r['task']
            scope = 'source_exposed_descriptive' if co == r['source'] else 'independent_of_posttraining_source'
            if task in ('AGE', 'SEX'):
                fam = 'demographic_SSL_' + ('source_exposed_SSL_probe' if co == r['source'] else 'external_to_SSL_source') + '_' + task
                add(rows, r, 'demographics_adaptation', path, i, family=fam, evaluation_scope=scope)
            elif task in ('DX', 'AD_DX'):
                add(rows, r, 'diagnosis_adaptation', path, i,
                    family='diagnosis_' + task + '_' + obj + '_' + scope, evaluation_scope=scope)
            else:
                task = canonical_task(task, co)
                rr = dict(r, task=task, section='rows')
                method = 'clinical+FM' if 'followup' in task else 'FM'
                add(rows, r, 'cognitive_adaptation', path, i, task=task,
                    family=adaptation_family(rr), analysis_method=method,
                    evaluation_scope=scope, protocol='sf100_exact_probe')

    # Original SI reference export omitted the separate clinical-increment rows.
    # Restore their archived raw tests, rather than treating missing exports as nulls.
    for model in MODELS[:4]:
        path = ROOT / 'runs/sample_alignment_v1/paper_completion_v1/inference' / ('frozen_' + model + '.json')
        for i, r in enumerate(read(path)['rows']):
            if not r['family'].startswith('clinical_increment_'):
                continue
            flat = {k: v for k, v in r.items() if k != 'statistics'}
            flat.update(r['statistics'])
            add(rows, flat, 'local_clinical_increment', path, i,
                objective='frozen', analysis_method='clinical+FM', protocol='original_model_atlas_probe')
    # The new report saved these HRV predictions but omitted their paired tests.
    # Only CPU inference is added; fitted models and outcomes remain unchanged.
    path = NEW / 'local_predictions.json'
    preds = read(path)
    signature = digest(path)
    cache = RUN / 'additional_recall_inference.json'
    if cache.exists():
        extra = json.loads(cache.read_text())
        if extra['predictions_sha256'] != signature:
            raise ValueError('Additional recall inference inputs changed')
    else:
        from paper_paired_stats import paired
        tests = []
        for r in preds:
            if r['task'] != 'MEM_delayed_recall_baseline_prediction' or r['method'] != 'clinical+FM':
                continue
            refs = [q for q in preds if q['model'] == r['model'] and q['cohort'] == r['cohort'] and q['task'] == r['task']]
            clinical = next(q for q in refs if q['source'] == r['source'] and q['objective'] == r['objective'] and q['target'] == r['target'] and q['method'] == 'clinical')
            comparisons = [('clinical', clinical)]
            if r['objective'] != 'frozen':
                comparisons.append(('frozen', next(q for q in refs if q['objective'] == 'frozen' and q['method'] == 'clinical+FM')))
            for refname, ref in comparisons:
                for field in ('subjects', 'scan_ids', 'y'):
                    if r[field] != ref[field]:
                        raise ValueError('Unpaired additional recall inference: ' + field)
                seed = int(hashlib.sha256(json.dumps([r['source'], r['model'], r['objective'], r['target'], refname]).encode()).hexdigest()[:8], 16)
                info = {k: r[k] for k in ('source', 'model', 'objective', 'target', 'cohort', 'task', 'method')}
                tests.append(dict(info, reference=refname,
                    **paired(r['y'], ref['prediction'], r['prediction'], r['subjects'], seed=seed)))
        extra = dict(predictions_sha256=signature, rows=tests)
        cache.write_text(json.dumps(extra, indent=2) + '\n')
    for i, r in enumerate(read(cache)['rows']):
        if r['reference'] == 'frozen':
            add(rows, r, 'cognitive_adaptation', cache, i, family=adaptation_family(r),
                section='rows', analysis_method='clinical+FM', protocol='original_clinical_recall_probe')
        else:
            fam = clinical_family(r['task'], r['objective'])
            add(rows, r, 'local_clinical_increment', cache, i, family=fam,
                analysis_method='clinical+FM', protocol='original_clinical_recall_probe')
    path, clinical_rows = original_adapted_clinical()
    for i, r in enumerate(clinical_rows):
        add(rows, r, 'local_clinical_increment', path, i, protocol='original_model_atlas_probe')

    # Additional original-only source-FC/GFM axis contrasts are not local probes.
    path = OLD / 'S1_SF100_paired.csv'
    for i, r in enumerate(read(path)):
        if r['family'].startswith('transfer_'):
            add(rows, r, 'frozen_source_axis_reference', path, i)
    path = OLD / 'S3_recall_axes_paired.csv'
    for i, r in enumerate(read(path)):
        add(rows, r, 'frozen_source_axis_reference', path, i)

    old_panels = [('S5_external_paired', 'cognitive_adaptation'),
                  ('S5_demographic_paired', 'demographics_adaptation'),
                  ('S5_diagnosis_paired', 'diagnosis_adaptation'),
                  ('S5_source_paired', 'source_cv'),
                  ('S2_all_paired', 'atlas_frozen_reference')]
    for filename, panel in old_panels:
        path = OLD / (filename + '.csv')
        for i, r in enumerate(read(path)):
            add(rows, r, panel, path, i,
                protocol='source_fold_cv' if panel == 'source_cv' else 'original_model_atlas_probe',
                objective='supervised' if panel == 'source_cv' else clean(r.get('objective'), 'frozen'),
                source=r.get('cohort') if panel == 'source_cv' else clean(r.get('source')),
                adaptation_target=r['task'] if panel == 'source_cv' else clean(r.get('adaptation_target')))

    path = NEW / 'source_cv_comparisons.json'
    for i, r in enumerate(read(path)):
        add(rows, r, 'source_cv', path, i, family='source_supervised_' + r['target'],
            cohort=r['source'], task=r['target'], objective='supervised',
            adaptation_target=r['target'], reference='frozen', candidate='adapted',
            protocol='source_fold_cv', evaluation_scope='independent_fold_specific_encoders')
    path = NEW / 'axes_comparisons.json'
    for i, r in enumerate(read(path)):
        add(rows, r, 'cognitive_adaptation', path, i, family=adaptation_family(r),
            section='axes', analysis_method='source_axis', protocol='source_only_axis')

    # Frozen LOSO and separate HRV recall: original rows are added only once.
    path = OLD / 'S3_LOSO_paired.csv'
    for i, r in enumerate(read(path)):
        add(rows, r, 'loso_frozen', path, i, task=r.get('domain', 'MEM_delayed_recall'))
    path = NEW / 'loso_comparisons.json'
    for i, r in enumerate(read(path)):
        if r.get('model') not in ADDED:
            continue
        if r['objective'] == 'frozen':
            add(rows, r, 'loso_frozen', path, i)
        else:
            add(rows, r, 'loso_adaptation', path, i,
                family='loso_' + r['task'] + '_' + r['objective'] + '_independent_of_posttraining_source')
    path = OLD / 'S5_loso_paired.csv'
    for i, r in enumerate(read(path)):
        add(rows, r, 'loso_adaptation', path, i)

    path = NEW / 'identification_comparisons.json'
    for i, r in enumerate(read(path)):
        if r.get('model') not in ADDED:
            add(rows, r, 'identification_frozen', path, i, model=r['candidate'],
                task=r['metric'], objective='frozen', protocol='scan_identification')
        elif r['objective'] == 'frozen':
            add(rows, r, 'identification_frozen', path, i, task=r['metric'], protocol='scan_identification')
        else:
            co = 'nacc' if r['cohort'] == 'nacc_repeated' else r['cohort']
            scope = 'source_exposed_descriptive' if co == r['source'] else 'independent_of_posttraining_source'
            add(rows, r, 'identification_adaptation', path, i, task=r['metric'],
                family='identification_' + r['metric'] + '_' + r['objective'] + '_' + scope,
                evaluation_scope=scope, protocol='scan_identification')
    path = OLD / 'S5_identification_paired.csv'
    for i, r in enumerate(read(path)):
        add(rows, r, 'identification_adaptation', path, i, protocol='scan_identification')

    path = OLD / 'S4_readout_paired.csv'
    for i, r in enumerate(read(path)):
        add(rows, r, 'readout', path, i, task=clean(r['endpoint'], 'identification'), protocol='readout_sensitivity')
    path = NEW / 'readout_comparisons.json'
    for i, r in enumerate(read(path)):
        fam = 'identification_' + r['metric'] if r['task'] == 'identification' else 'cognition_' + r['task']
        add(rows, r, 'readout', path, i, family=fam, reference='native', protocol='readout_sensitivity')

    path = OLD / 'S7_fusion_paired.csv'
    for i, r in enumerate(read(path)):
        add(rows, r, 'fusion', path, i, protocol='dimension_controlled_fusion')
    path = NEW / 'fusion_comparisons.json'
    for i, r in enumerate(read(path)):
        ep = outcome(r['task'])
        if 'followup' in r['task']:
            group = {'clinical+FC': 'dimension_matched', 'clinical': 'clinical_reference',
                     'clinical+FC5': 'additive_sensitivity'}[r['reference']]
            fam = 'fusion_' + group + '_followup_' + ep
        else:
            fam = 'fusion_dimension_matched_observational_' + ep
        add(rows, r, 'fusion', path, i, family=fam,
            task=canonical_task(r['task'], r['cohort']), protocol='dimension_controlled_fusion')

    for path in (OLD / 'S7_robustness_paired.csv', NEW / 'robustness_comparisons.json'):
        for i, r in enumerate(read(path)):
            atlas = 'sch450' if r['model'] in ('brainjepa', *ADDED) else 'a424'
            reference_only = r['candidate'] == 'adjusted+FC' or (r['reference'] == 'FC_primary' and r['candidate'] == 'FC_short')
            add(rows, r, 'robustness', path, i, task=r['endpoint'], atlas=atlas,
                reference_only=reference_only,
                model='shared_' + atlas + '_FC' if reference_only else r['model'],
                protocol='reference_only' if reference_only else ('new_TR2_context' if r['model'] in ADDED else 'original_native_TR_context'))
    # This report already includes original and added-model raw calibration tests.
    path = NEW / 'harmonization_comparisons.json'
    for i, r in enumerate(read(path)):
        add(rows, r, 'alignment', path, i, protocol='embedding_alignment')
    path = NEW / 'paired_visit/paired_inference.csv'
    for i, r in enumerate(read(path)):
        add(rows, r, 'paired_visit', path, i, protocol='paired_visit_5PC')

    # Same statistical question, distinct encoder/window protocols remain labeled.
    paths = (OLD / 'S6_crosscohort_frozen_paired.csv',
             ROOT / 'runs/new_fm_window_reencoding_v1/reports/all/paired_comparisons.csv')
    for path in paths:
        original = path.parent == OLD
        for i, r in enumerate(read(path)):
            obj = 'frozen' if original else r['objective']
            ref = r['reference']
            refclass = 'FC' if ref.endswith('_FC') else ref
            # Original scan-mean tests corrected all reference classes together.
            # Do not narrow that family after seeing results.
            fam = ('physiology_scan_mean_' + obj if r['scope'] == 'scan_mean'
                   else 'physiology_' + r['scope'] + '_' + obj + '_' + refclass)
            add(rows, r, 'physiology', path, i, family=fam, objective=obj,
                protocol='contextual_temporal_tokens' if original else 'independent_window_reencoding',
                task='heart_or_pulse_rate', reference_class=refclass)
    return rows


def deduplicate(rows):
    """Remove duplicate hypotheses, retaining the first archived computation.

    Shared atlas-only controls are the same hypothesis repeated under FM labels.
    Their scores must agree before reuse; different Monte Carlo P estimates are
    retained in an audit but do not create additional hypotheses.
    """
    retained, removed, seen = [], [], {}
    for r in rows:
        uid = r['hypothesis_id']
        if uid in seen:
            previous = seen[uid]
            for k in ('reference_score', 'candidate_score', 'n'):
                a, b = previous.get(k), r.get(k)
                if a is not None and b is not None and not np.isclose(a, b, atol=1e-12, rtol=0):
                    raise ValueError(f'Duplicate identity with unequal {k}: {uid}')
            removed.append(dict(r, retained_input=previous['input_file_provenance'],
                                retained_row=previous['input_row'], reason='same_hypothesis_same_scores_first_archived_test_retained'))
        else:
            seen[uid] = r
            retained.append(r)
    return retained, removed


def adjust(rows):
    f = pd.DataFrame(rows)
    f['holm_p_six_models'] = np.nan
    for fam, ix in f.groupby('family', sort=True).groups.items():
        f.loc[ix, 'holm_p_six_models'] = holm(f.loc[ix, 'raw_p'].tolist())
        f.loc[ix, 'family_size_six_models'] = len(ix)
    f['holm_p_whole_release_sensitivity'] = holm(f.raw_p.tolist())
    f['significant_six_models'] = f.holm_p_six_models.le(.05)
    f['direction'] = np.where(f.delta.gt(1e-12), 'positive', np.where(f.delta.lt(-1e-12), 'negative', 'zero'))
    f['status_six_models'] = np.where(f.significant_six_models, f.direction + '_corrected', 'not_detected')
    f['previously_significant'] = pd.to_numeric(f.archived_holm_p, errors='coerce').le(.05)
    f['correction_changed_decision'] = f.archived_holm_p.notna() & f.previously_significant.ne(f.significant_six_models)
    return f


def consistency(f):
    """Descriptive replication without treating seeds or variants as cohorts."""
    x = f[f.model.isin(MODELS)].copy()
    # HRV pre/post are not independent cohorts. Retain contexts in task labels.
    x['replication_cohort'] = x.cohort.replace({'hrv_pre': 'hrv', 'hrv_post': 'hrv', 'nacc_repeated': 'nacc'})
    # Pooling across sources, objectives, input modes or readouts is forbidden.
    keys = ['panel', 'model', 'source', 'objective', 'adaptation_target', 'task',
            'analysis_method', 'reference', 'candidate', 'mode', 'setting', 'scope',
            'axis_source', 'contrast', 'protocol', 'evaluation_scope', 'metric', 'family']
    # References containing a model name are normalized without merging models.
    x['reference'] = x.reference.replace({m + '_frozen': 'frozen' for m in MODELS})
    # Frozen candidate names differ across rows; one model is already a key.
    x['candidate'] = np.where(x.objective.eq('frozen') & x.candidate.eq(x.model), 'FM', x.candidate)
    summaries = []
    for vals, g in x.groupby(keys, dropna=False, sort=True):
        cohorts = []
        for co, a in g.groupby('replication_cohort'):
            ds = a.delta.to_numpy(float)
            cohorts.append(dict(cohort=co, positive=bool((ds > 1e-12).all()),
                                negative=bool((ds < -1e-12).all()),
                                corrected_gain=bool(((a.delta > 0) & a.significant_six_models).any()),
                                corrected_loss=bool(((a.delta < 0) & a.significant_six_models).any())))
        n = len(cohorts)
        pos = sum(a['positive'] for a in cohorts)
        neg = sum(a['negative'] for a in cohorts)
        wins = sum(a['corrected_gain'] for a in cohorts)
        losses = sum(a['corrected_loss'] for a in cohorts)
        external = str(dict(zip(keys, vals))['evaluation_scope']) not in (
            'source_exposed_descriptive', 'source_exposed_SSL_probe')
        label = 'single_cohort_only' if n < 2 else 'mixed_or_unresolved'
        if n >= 2 and pos == n:
            label = 'directionally_consistent_positive'
        if n >= 2 and neg == n:
            label = 'directionally_consistent_negative'
        if n >= 2 and wins >= 2 and losses == 0 and external:
            label = 'replicated_external_gain'
            if pos == n:
                label = 'consistent_external_gain'
        if n >= 2 and losses >= 2 and wins == 0:
            label = 'replicated_loss'
            if neg == n:
                label = 'consistent_loss'
        summaries.append(dict(zip(keys, vals), n_cohorts=n, positive_cohorts=pos,
                              negative_cohorts=neg, corrected_gain_cohorts=wins,
                              corrected_loss_cohorts=losses, label=label,
                              cohorts=';'.join(sorted(a['cohort'] for a in cohorts)),
                              delta_min=g.delta.min(), delta_max=g.delta.max()))
    return pd.DataFrame(summaries)


def across_models(f):
    """Configuration-level concordance, explicitly not independent replication."""
    x = f[f.model.isin(MODELS)].copy()
    x['reference_group'] = x.reference.replace({m + '_frozen': 'frozen' for m in MODELS})
    x['candidate_group'] = np.where(x.candidate.eq(x.model), 'FM', x.candidate)
    keys = ['panel', 'cohort', 'source', 'objective', 'adaptation_target', 'task',
            'analysis_method', 'reference_group', 'candidate_group', 'mode', 'setting', 'scope', 'contrast', 'metric', 'family']
    result = []
    for vals, g in x.groupby(keys, dropna=False, sort=True):
        per = g.groupby('model').agg(positive=('delta', lambda a: bool((a > 1e-12).all())),
                                    negative=('delta', lambda a: bool((a < -1e-12).all())))
        if len(per) < 2:
            continue
        result.append(dict(zip(keys, vals), models_observed=len(per),
                           models_positive=int(per.positive.sum()), models_negative=int(per.negative.sum()),
                           corrected_gain_models=g.loc[g.significant_six_models & g.delta.gt(0), 'model'].nunique(),
                           corrected_loss_models=g.loc[g.significant_six_models & g.delta.lt(0), 'model'].nunique(),
                           complete_six_model_coverage=len(per) == 6,
                           interpretation='Configuration concordance; models are not independent cohorts'))
    return pd.DataFrame(result)


def policy():
    return [
        ('Multiplicity', 'Holm alpha=0.05 within each named scientific comparison family across all available six-model contrasts, retaining FC/GFM references where originally part of that family. This is not six separate within-model corrections. A whole-release Holm column is provided only as a more conservative sensitivity analysis.'),
        ('Retrospective status', 'Family harmonization and consistency terminology were specified after viewing results. These are exploratory reporting rules, not prospectively registered confirmatory criteria.'),
        ('Family mapping', 'Original scientific family definitions are retained where possible. Added-model aliases map into those families. Supervised MEM and EF adaptation targets are combined when the original family combined them. Model names are never used to split a family. Distinct reference/input/protocol contrasts retain their identities.'),
        ('Raw tests', 'No models are retrained. Raw paired tests and pointwise confidence intervals are reused. When valid-swap and exceedance counts are supplied, P=(1+b)/(1+B) is reconstructed. Missing/nonestimable planned tests stay in a family with P=1.'),
        ('Duplicates', 'Already merged exports are not appended twice. Identical atlas-only FC controls repeated under model labels are counted once, retaining the first original computation after checking equal scores and N. The duplicate ledger preserves discarded Monte Carlo results.'),
        ('Directionally consistent', 'For the same model, source, objective, target, task, reference, input mode and evaluation setting, the effect has the same nonzero direction in every eligible evaluated cohort, with at least two cohorts. This is descriptive and does not establish significance.'),
        ('Replicated external gain', 'The same fixed configuration improves after joint Holm correction in at least two independent downstream cohorts, with no corrected loss in another cohort. Other cohorts may be inconclusive. This is an exploratory replication label, not a formal meta-analysis.'),
        ('Consistent external gain', 'Replicated external gain plus a positive point-estimate difference in every eligible evaluated cohort. Absolute candidate performance and effect magnitude must also be shown. Recovery from a negative baseline is not automatically clinically useful.'),
        ('No detected incremental benefit', 'Zero corrected gains is reported as no improvement detected in this panel, never equivalence, consistent absence of information, or proof of no benefit. No clinical equivalence margin was specified.'),
        ('Independence', 'Repeated scans, seeds, objectives, readouts and input modes do not count as independent replication. HRV pre/post and NACC repeated samples count as the same parent cohort. Shared training cohorts in LOSO also limit independence of target results.'),
        ('Across-model concordance', 'Agreement among six configurations on the same participants is reported separately from across-cohort replication. Models share data and are not six independent replications.'),
        ('Coverage', 'All available tests are corrected; missing tasks are not imputed as null effects. Source-exposed SSL probes are descriptive. Historical ROI-gain/random/GRU/diagnostic-objective controls remain outside the six-model panel.'),
        ('Physiology', 'Native temporal-token and independent-window reencoding protocols remain explicitly labeled. Frozen tests share broader scope-by-reference-class correction across six models, but this does not turn the protocols into a matched head-to-head experiment. Older adapted physiology uses a distinct historical protocol and is not relabeled as a six-model comparison.'),
        ('Input robustness', 'The original >=200-native-frame subset is not proof of no padding after new-model TR2 resampling. Acquisition/input sensitivity does not isolate a causal effect of TR or duration.'),
    ]


def coverage(f):
    gaps = [
        ('atlas_frozen_reference', 'Original atlas-matched FC/FC-Tucker local reference panel has four models; the two new models have SF100 local comparisons and static-FC fusion comparisons, not the full same-panel Tucker extension.'),
        ('loso_frozen', 'HRV-ER one-way external delayed-recall LOSO has original-model results; added-model LOSO artifacts cover MEM in four cohorts and EF in five, not HRV delayed recall.'),
        ('local_clinical_increment', 'Missing original-model adapted-versus-clinical and added-model HRV recall tests were completed from existing seed-averaged predictions, without retraining. Six-model families include the original four models and the new two models; source-specific probe implementations remain labeled.'),
        ('cognitive_adaptation', 'The omitted new-model HRV-ER clinical+FM recall paired tests were computed from existing predictions in this release; no encoder or probe was retrained.'),
        ('physiology', 'Only four models have native temporal-token analysis; two use independent-window reencoding. Adapted six-model physiology is not one matched protocol.'),
        ('readout', 'Alternative readouts are architecture-specific. BrainLM13 layer-3 embedding-connectivity controls exist only for that model and retain separate layer/readout families.'),
    ]
    table = f.groupby(['panel', 'model'], dropna=False).size().unstack(fill_value=0).reset_index()
    return table, pd.DataFrame(gaps, columns=['panel', 'limitation'])








def validate(f, families, rows):
    assert f.hypothesis_id.is_unique
    assert f.raw_p.between(0, 1).all()
    for _, g in f.groupby('family'):
        # Independent direct formula: adjusted p_i=max_{p_j<=p_i}
        # (m-rank_j+1)*p_j. This does not call the production Holm function.
        p = g.raw_p.to_numpy()
        order = np.argsort(p)
        expect = np.empty(len(p))
        for rank, index in enumerate(order):
            expect[index] = min(1., max((len(p) - j) * p[order[j]] for j in range(rank + 1)))
        np.testing.assert_allclose(g.holm_p_six_models, expect, atol=1e-15, rtol=0)
        assert g.family_size_six_models.eq(len(g)).all()
    # Every six-model-eligible family must have combined original/added models.
    for panel in ['source_cv', 'readout', 'fusion', 'diagnosis_adaptation', 'demographics_adaptation',
                  'identification_adaptation', 'loso_adaptation', 'cognitive_adaptation', 'paired_visit']:
        assert set(f.loc[f.panel.eq(panel), 'model']) >= set(MODELS), panel
    for p, sha in INPUTS.items():
        assert digest(ROOT / p) == sha, 'Input changed: ' + p
    assert len(f) == sum(families.n_tests)
    (RUN / 'validation.json').write_text(json.dumps(dict(status='passed',
        contrasts=len(f), families=len(families),
        independent_holm='Independent direct Holm formula matches every family',
        source_files_unchanged=len(INPUTS),
        checks=['unique hypothesis IDs', 'finite raw P reconstruction', 'family sizes',
                'six-model panel membership', 'input SHA256 integrity']), indent=2) + '\n')



def atlas_family(task,co):
    setting='hrv_recall' if 'delayed_recall' in task else 'followup' if 'followup' in task else 'observational' if co in ('adni','nacc') else 'intervention_baseline'
    return 'frozen_reference_'+setting+'_'+outcome(task)
