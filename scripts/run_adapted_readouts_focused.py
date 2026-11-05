#!/usr/bin/env python3
"""Bounded existing-adapter readout supplement; original full results are immutable.

Retain native and final-layer connectivity only, external ADNI/NACC cognition,
and seven-cohort identification. Reuse the registered numerical operators,
samples, splits, three adapter seeds, and inference budgets from v1.
"""
from pathlib import Path
import sys
import json
import os
import shutil
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import run_adapted_readouts as base
from fm_matrix.data import save, sha

OLD = base.RUN
RUN = ROOT/'runs/adapted_readouts_focused_v1'
OUT = ROOT/'results/generated/adapted_readouts_focused_v1'
READOUTS = ('native', 'connectivity')
_original_names = base.frozen.readout_names
_original_last = base.last_readouts
_original_features = base.feature_set
_original_predict = base.predict
_original_outputs = base.expected_outputs
_original_jobs = base.jobs
_original_inference = base.inference


def selected_last(net, model, x):
    # Compute exactly the registered operators from one final-layer forward pass.
    base.frozen.readout_names = _original_names
    try:
        values = _original_last(net, model, x)
    finally:
        base.frozen.readout_names = lambda model: list(READOUTS)
    return {k: values[k] for k in READOUTS}


def selected_features(s, co):
    values = _original_features(s, co)
    return {k: values[k] for k in READOUTS}


def selected_predict(s, co, ep, device):
    if co != s[0]:
        return _original_predict(s, co, ep, device)


def selected_outputs(s, co):
    return [p for p in _original_outputs(s, co)
            if not (co == s[0] and p.parent.name == 'cognition')]


def selected_jobs(d):
    return [j for j in _original_jobs(d)
            if not ('endpoint' in j and j['cohort'] == j['state'][0])]


def relabel_inference(d):
    d['rows'] = [r for r in d['rows']
                 if r['candidate'] in READOUTS and r['reference'] in (*READOUTS, 'sf100_fc')]
    for r in d['rows']:
        r['family'] = r['family'].replace('adapted_readouts::', 'adapted_readouts_focused::')
    d['scores'] = [r for r in d['scores'] if r['readout'] in READOUTS]
    if 'ensemble_predictions' in d:
        d['ensemble_predictions'] = {k: d['ensemble_predictions'][k] for k in READOUTS}
    d['scope_revision'] = 'Post hoc computational scope reduction; exploratory, not confirmatory.'
    return d


def selected_inference(s, co, ep=None):
    _original_inference(s, co, ep)
    p = RUN/'inference'/base.group_id(s)/(co+('_'+ep if ep else '')+'.json')
    save(p, relabel_inference(base.read(p)))


base.RUN, base.OUT = RUN, OUT
base.frozen.readout_names = lambda model: list(READOUTS)
base.last_readouts = selected_last
base.feature_set = selected_features
base.predict = selected_predict
base.expected_outputs = selected_outputs
base.jobs = selected_jobs
base.inference = selected_inference
# Re-export the engine API; its functions resolve the patched module globals.
for _name in dir(base):
    if not _name.startswith('_') and _name not in globals():
        globals()[_name] = getattr(base, _name)
# Names already defined above must resolve to the selected API as well.
feature_set = selected_features
last_readouts = selected_last
predict = selected_predict
expected_outputs = selected_outputs
jobs = selected_jobs
inference = selected_inference


def verify():
    d = base.read(RUN/'manifest.json')
    for key, path in [('code_sha256', Path(base.__file__)),
                      ('frozen_engine_sha256', Path(base.frozen.__file__)),
                      ('focused_engine_sha256', Path(__file__))]:
        if d[key] != sha(path):
            raise ValueError('Registered code changed: '+key)
    return d


def prepare():
    parent = base.read(OLD/'manifest.json')
    assert parent['code_sha256'] == sha(Path(base.__file__))
    assert parent['frozen_engine_sha256'] == sha(Path(base.frozen.__file__))
    d = dict(parent)
    d.update(focused_engine_sha256=sha(Path(__file__)),
             parent_manifest_sha256=sha(OLD/'manifest.json'),
             readouts={m: list(READOUTS) for m in base.MODELS},
             scope='All nine configurations, both sources, MEM/EF/SSL, seeds42/43/44. External ADNI/NACC local cognitive probes only; seven-cohort identification. Native versus final-layer connectivity.',
             source_evaluation='No new source-cohort cognitive fits. Source-exposed identification remains descriptive, separately corrected.',
             inference='Original 5000 participant bootstraps and 20000 swaps; Holm across the complete bounded panel using original task/metric/mode/objective/exposure/contrast grouping. Families are exploratory and were narrowed after partial full-panel results existed.',
             revision_reason='User-authorized computational scope reduction, uniform across models, cohorts and objectives; no readout selected by target performance.',
             old_outputs='Preserved separately at runs/adapted_readouts_v1 and results/generated/adapted_readouts_v1; do not mix their coverage/correction with this panel.',
             automatic_shutdown=False)
    path = RUN/'manifest.json'
    if path.exists() and base.read(path) != json.loads(json.dumps(d)):
        raise ValueError('Focused manifest already registered differently')
    save(path, d)
    counts = dict(cognitive_outputs=0, identity_outputs=0, prediction_parts=0, feature_cohorts=0, inference_outputs=0)
    for r in d['states']:
        s = r['state']; old = OLD/'states'/r['id']; dest = base.directory(s)
        target = 'nacc' if s[0] == 'adni' else 'adni'
        for ep in ('MEM', 'EF'):
            p = old/'cognition'/f'{target}_{ep}.json'
            if p.exists():
                result = base.read(p)
                for key in ('predictions', 'predictions_by_seed', 'scores'):
                    result[key] = {k: result[key][k] for k in READOUTS}
                result['reused_from'] = str(p); result['reused_sha256'] = sha(p)
                save(dest/'cognition'/p.name, result); counts['cognitive_outputs'] += 1
            for name in READOUTS:
                p = old/'prediction_parts'/f'{target}_{ep}_{name}.json'
                if p.exists():
                    q = dest/'prediction_parts'/p.name; q.parent.mkdir(parents=True, exist_ok=True)
                    if not q.exists(): shutil.copy2(p, q)
                    counts['prediction_parts'] += 1
        for co in base.frozen.PAIRS:
            p = old/'identity'/(co+'.json')
            if not p.exists(): continue
            result = base.read(p); src = p.with_suffix('.npz')
            if sha(src) != result['similarity_sha256']: raise ValueError('Old identity hash mismatch')
            q = dest/'identity'/src.name; q.parent.mkdir(parents=True, exist_ok=True)
            with np.load(src) as z:
                arrays = {f'{mode}__{k}': z[f'{mode}__{k}'] for mode in ('native', 'projected128') for k in READOUTS}
                tmp = q.with_suffix('.tmp.npz'); np.savez(tmp, **arrays); tmp.replace(q)
            result.update(similarity_sha256=sha(q), reused_from=str(p), reused_sha256=sha(p))
            save(q.with_suffix('.json'), result); counts['identity_outputs'] += 1
        for co in base.frozen.COHORTS:
            folder = old/'scratch'/co
            if not (folder/'complete.json').exists() or (folder/'evicted.json').exists(): continue
            if all(p.exists() for p in selected_outputs(s, co)): continue
            result = base.read(folder/'complete.json'); target_folder = dest/'scratch'/co
            target_folder.mkdir(parents=True, exist_ok=True)
            for k in READOUTS:
                p = folder/(k+'.npy')
                if sha(p) != result['features_sha256'][k]: raise ValueError('Old feature hash mismatch')
                q = target_folder/p.name
                if not q.exists(): os.link(p, q)
            result['features_sha256'] = {k: result['features_sha256'][k] for k in READOUTS}
            result['reused_from'] = str(folder/'complete.json')
            save(target_folder/'complete.json', result); counts['feature_cohorts'] += 1
        if all(p.exists() for co in base.frozen.COHORTS for p in selected_outputs(s, co)):
            save(dest/'complete.json', dict(status='complete', state=s, reused_from=str(old), new_training=False))
    for j in selected_jobs(d):
        if j['action'] != 'inference': continue
        p = OLD/'inference'/base.group_id(j['state'])/(j['cohort']+('_'+j['endpoint'] if 'endpoint' in j else '')+'.json')
        if p.exists():
            result = relabel_inference(base.read(p))
            result['reused_from'] = str(p); result['reused_sha256'] = sha(p)
            save(Path(j['output']), result); counts['inference_outputs'] += 1
    save(RUN/'reuse_audit.json', dict(**counts, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(json.dumps(dict(**counts, **base.report(d))), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare']: prepare()
    elif sys.argv[1:] == ['report']: print(base.report(verify()))
    else: raise SystemExit('Use prepare/report, or schedule_adapted_readouts_focused.py for workers')
