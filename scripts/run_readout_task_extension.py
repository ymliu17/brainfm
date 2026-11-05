#!/usr/bin/env python3
"""CPU-only frozen final-layer readout extension: demographics and diagnosis.

Reuses immutable SF100 samples, outcomes, outer/inner folds and FC predictions.
No encoder inference, new splits, readout selection, manuscript edits or shutdown.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import run_connectivity_readouts as frozen
import sf100_benchmark as sf
from benchmark_evaluation import predict_prepared, score
from fm_matrix.data import save, sha
from paper_paired_stats import paired, holm

RUN = ROOT/'runs/readout_task_extension_v1'
OUT = ROOT/'results/generated/readout_task_extension_v1'
TASKS = {'adni': ('AGE', 'SEX', 'DX'), 'nacc': ('AGE', 'SEX', 'DX', 'AD_DX')}


def read(path):
    return json.loads(Path(path).read_text())


def jobs():
    return [dict(id=f'{m}/{co}/{ep}', model=m, cohort=co, task=ep,
                 output=f'evaluation/{m}/{co}/{ep}/complete.json')
            for co, eps in TASKS.items() for ep in eps for m in frozen.MODELS]


def audit_template(old, metadata, endpoint):
    if old['status'] != 'complete' or old.get('covariates'):
        raise ValueError('Expected complete imaging-only parent task')
    ix = metadata.set_index('scan').index.get_indexer(old['scan_ids'])
    g = np.asarray(old['subjects']).astype(str)
    y = np.asarray(old['y'], dtype=float)
    if (ix < 0).any() or metadata.iloc[ix].subject.astype(str).tolist() != g.tolist():
        raise ValueError('Parent sample/order mismatch')
    col = {'AGE': 'age', 'SEX': 'sex'}.get(endpoint, endpoint)
    np.testing.assert_array_equal(metadata.iloc[ix][col].to_numpy(float), y)
    if endpoint != 'AGE' and set(np.unique(y)) != {0., 1.}:
        raise ValueError('Expected binary classification')
    audits = old['split_audit']
    if len(audits) != 15:
        raise ValueError('Expected three seeds and five outer folds')
    for seed in (0, 1, 2):
        seen = np.zeros(len(y), dtype=int)
        folds = [f for f in audits if f['seed'] == seed]
        if len(folds) != 5:
            raise ValueError('Incorrect fold count')
        for f in folds:
            tr, te, it, iv = [np.asarray(f[k], int) for k in
                              ('train', 'test', 'inner_train', 'validation')]
            if set(tr) & set(te) or set(tr) | set(te) != set(range(len(y))):
                raise ValueError('Invalid outer partition')
            if set(it) & set(iv) or set(it) | set(iv) != set(tr):
                raise ValueError('Invalid inner partition')
            if set(g[tr]) & set(g[te]) or set(g[it]) & set(g[iv]):
                raise ValueError('Participant leakage')
            seen[te] += 1
        if not (seen == 1).all():
            raise ValueError('OOF rows must be tested exactly once per seed')
    return ix, y, g


def register():
    paths = [Path(__file__), ROOT/'scripts/run_connectivity_readouts.py',
             ROOT/'src/benchmark_evaluation.py', ROOT/'src/paper_paired_stats.py',
             ROOT/'src/paper_completion.py', frozen.RUN/'manifest.json']
    counts = []
    for co, eps in TASKS.items():
        metadata = sf.labels(co)
        for ep in eps:
            p = sf.RUN/'evaluation'/co/(ep+'.json')
            old = read(p)
            _, y, g = audit_template(old, metadata, ep)
            paths.append(p)
            counts.append(dict(cohort=co, task=ep, scans=len(y), participants=len(set(g))))
        for model in frozen.MODELS:
            stamp = frozen.RUN/'features'/model/co/'complete.json'
            d = read(stamp)
            if d['scan_ids'] != metadata.scan.tolist():
                raise ValueError('Feature rows differ from task metadata')
            if not set(frozen.readout_names(model)) <= set(d['features_sha256']):
                raise ValueError('Missing required final-layer readout')
            paths.append(stamp)
    manifest = dict(version=1, jobs=jobs(), counts=counts,
        input_sha256={str(p): sha(p) for p in paths},
        readouts={m: frozen.readout_names(m) for m in frozen.MODELS},
        automatic_shutdown=False, encoder_training=False, gpu_required=False,
        protocol='Immutable SF100 samples, labels, 15 participant-grouped outer/inner partitions; training-only StandardScaler/exact PCA<=100; same ridge/logistic grid. Average OOF predictions across seeds0/1/2 before scoring.',
        inference='5000 participant-cluster bootstraps and 20000 two-sided conditional prediction swaps; session-weighted scores. No new absolute-null tests or model refits during resampling.',
        families='Holm across all nine configurations, eligible cohorts and all final-layer alternative readouts, separately by endpoint and comparator (primary readout or Schaefer100 FC). AD_DX remains a distinct diagnostic endpoint. No historical intermediate-layer controls. No final adjusted P values until the full registered panel completes.',
        scope='Frozen AGE/SEX/DX in ADNI/NACC plus NACC AD_DX only; additive sensitivity, no replacement of primary results or manuscript edits.')
    p = RUN/'manifest.json'
    if p.exists() and read(p) != manifest:
        raise ValueError('Registered protocol or inputs changed')
    if not p.exists():
        save(p, manifest)
    return manifest


def verify():
    manifest = read(RUN/'manifest.json')
    for p, h in manifest['input_sha256'].items():
        if sha(Path(p)) != h:
            raise ValueError('Registered dependency changed: '+p)
    return manifest


def family(endpoint, reference):
    return f'frozen_readout_extension::{endpoint}::{reference}'


def evaluate(model, co, ep):
    verify()
    if ep not in TASKS[co]:
        raise ValueError('Unregistered endpoint')
    dest = RUN/'evaluation'/model/co/ep
    if (dest/'complete.json').exists():
        return
    start = time.time()
    template = sf.RUN/'evaluation'/co/(ep+'.json')
    old = read(template)
    ix, y, g = audit_template(old, sf.labels(co), ep)
    stamp_path = frozen.RUN/'features'/model/co/'complete.json'
    stamp = read(stamp_path)
    names = frozen.readout_names(model)
    pp, byseed, dims, timings = {}, {}, {}, {}
    for name in names:
        cache = dest/'predictions'/(name+'.json')
        if cache.exists():
            d = read(cache)
            pp[name], byseed[name], dims[name] = d['prediction'], d['by_seed'], d['dimension']
            timings[name] = d['elapsed_seconds']
            continue
        path = stamp_path.parent/(name+'.npy')
        if sha(path) != stamp['features_sha256'][name]:
            raise ValueError('Feature hash mismatch: '+str(path))
        x = np.load(path, mmap_mode='r')
        tick = time.time()
        prep = {s: [] for s in (0, 1, 2)}
        for f in old['split_audit']:
            tr, te, it, iv = [np.asarray(f[k], int) for k in
                              ('train', 'test', 'inner_train', 'validation')]
            split_file = dest/'projections'/name/f"{f['seed']}_{f['fold']}.npz"
            if split_file.exists():
                with np.load(split_file) as z:
                    a, b, ai, bi = [z[k] for k in ('a', 'b', 'ai', 'bi')]
            else:
                a, b = frozen.project_indices(x, ix[tr], ix[te], 100, 'cpu')
                ai, bi = frozen.project_indices(x, ix[it], ix[iv], 100, 'cpu')
                split_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = split_file.with_suffix('.tmp.npz')
                np.savez(tmp, a=a, b=b, ai=ai, bi=bi)
                tmp.replace(split_file)
            prep[f['seed']].append((tr, te, (a, b), it, iv, (ai, bi)))
            save(dest/'progress.json', dict(readout=name, seed=f['seed'], fold=f['fold'],
                elapsed_seconds=time.time()-start, updated_utc=utc()))
        byseed[name] = [predict_prepared(prep[s], y, binary=ep!='AGE') for s in (0, 1, 2)]
        pp[name] = np.mean(byseed[name], axis=0)
        dims[name] = int(x.shape[1])
        timings[name] = time.time()-tick
        save(cache, dict(prediction=pp[name], by_seed=byseed[name], dimension=dims[name],
            elapsed_seconds=timings[name], feature_sha256=stamp['features_sha256'][name],
            template_sha256=sha(template)))
        print('prediction', model, co, ep, name, score(y, pp[name], ep!='AGE'),
              'seconds', round(timings[name], 1), flush=True)
        del x, prep
    pp['sf100_fc'] = old['predictions']['sf100_fc']
    metric = 'rho' if ep == 'AGE' else 'auc'
    rows = []
    for name in names:
        if name == 'native':
            continue
        for reference in ('native', 'sf100_fc'):
            cache = dest/'inference'/(reference+'_'+name+'.json')
            if cache.exists():
                row = read(cache)
            else:
                stats = paired(y, pp[reference], pp[name], g, metric=metric,
                               seed=frozen.seed(model+co+ep+reference+name))
                row = dict(model=model, cohort=co, task=ep, reference=reference,
                           candidate=name, family=family(ep, reference), **stats)
                save(cache, row)
            rows.append(row)
    save(dest/'complete.json', dict(status='complete', model=model, cohort=co, task=ep,
        rows=rows, scan_ids=old['scan_ids'], subjects=g, y=y, predictions=pp,
        predictions_by_seed=byseed, scores={n: score(y, p, ep!='AGE') for n, p in pp.items()},
        metric=metric, split_audit=old['split_audit'], template_sha256=sha(template),
        feature_dimensions=dims, elapsed_seconds=time.time()-start, readout_seconds=timings))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def report():
    manifest = read(RUN/'manifest.json')
    coverage, rows, scores = [], [], []
    for j in manifest['jobs']:
        p = RUN/j['output']
        coverage.append(dict(**j, status='complete' if p.exists() else 'pending'))
        if not p.exists():
            continue
        d = read(p)
        rows.extend(d['rows'])
        for n, s in d['scores'].items():
            scores.append(dict(model=j['model'], cohort=j['cohort'], task=j['task'],
                readout=n, score=s, metric=d['metric'], n=len(d['y']),
                participants=len(set(d['subjects']))))
    complete = all(c['status'] == 'complete' for c in coverage)
    OUT.mkdir(parents=True, exist_ok=True)
    cov = pd.DataFrame(coverage)
    cov.to_csv(OUT/'Coverage.csv', index=False)
    comparisons = pd.DataFrame(rows)
    if len(comparisons):
        comparisons['holm_p'] = np.nan
        if complete:
            for _, ix in comparisons.groupby('family').groups.items():
                comparisons.loc[ix, 'holm_p'] = holm(comparisons.loc[ix, 'permutation_p'].fillna(1).tolist())
        comparisons['inference_status'] = 'final_nine_configuration_family' if complete else 'pending_complete_panel'
        comparisons.to_csv(OUT/'Comparisons.csv', index=False)
        pd.DataFrame(scores).to_csv(OUT/'Scores.csv', index=False)
        # Standalone workbook, never embed large tables in the manuscript.
        temp = OUT/'Readout_demographics_diagnosis.tmp.xlsx'
        with pd.ExcelWriter(temp, engine='xlsxwriter') as w:
            comparisons.to_excel(w, sheet_name='Comparisons', index=False)
            pd.DataFrame(scores).to_excel(w, sheet_name='Scores', index=False)
            cov.to_excel(w, sheet_name='Coverage', index=False)
            pd.DataFrame(manifest['counts']).to_excel(w, sheet_name='Samples', index=False)
            pd.DataFrame([dict(item=k, value=manifest[k]) for k in
                          ('protocol', 'inference', 'families', 'scope')]).to_excel(w, sheet_name='Protocol', index=False)
        temp.replace(OUT/'Readout_demographics_diagnosis.xlsx')
    summary = dict(total=len(coverage), completed=sum(c['status']=='complete' for c in coverage),
                   all_complete=complete, automatic_shutdown=False, updated_utc=utc())
    save(RUN/'summary.json', summary)
    if complete:
        save(RUN/'complete.json', summary)
    return summary


def run(workers):
    manifest = register()
    lock = open(RUN/'runner.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active, failed = {}, set()
    logs = ROOT/'logs/readout_task_extension_v1'
    logs.mkdir(parents=True, exist_ok=True)
    # Small NACC diagnostic tasks finish early; no dependency on GPU work.
    ordered = sorted(manifest['jobs'], key=lambda j: (j['cohort']!='nacc',
        j['task'] not in ('DX', 'AD_DX'), j['model'] not in ('semantoks', 'brainlm111')))
    while True:
        changed = False
        for key, (proc, j, fh) in list(active.items()):
            if proc.poll() is None:
                continue
            fh.close()
            del active[key]
            if proc.returncode or not (RUN/j['output']).exists():
                failed.add(key)
            changed = True
            print('finished', key, proc.returncode, flush=True)
        done = {j['id'] for j in ordered if (RUN/j['output']).exists()}
        for j in ordered:
            if len(active) >= workers:
                break
            if j['id'] in done | set(active) | failed:
                continue
            fh = open(logs/(j['id'].replace('/', '_')+'.log'), 'a')
            cmd = [sys.executable, str(Path(__file__).resolve()), 'evaluate', '--model', j['model'],
                   '--cohort', j['cohort'], '--task', j['task']]
            active[j['id']] = (subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), j, fh)
            print('started', j['id'], flush=True)
        save(RUN/'status.json', dict(total=len(ordered), completed=len(done), running=list(active),
            failed=sorted(failed), pending=len(ordered)-len(done)-len(active)-len(failed),
            cpu_workers=workers, threads_per_worker=1, automatic_shutdown=False, updated_utc=utc()))
        if changed:
            report()
        if not active:
            break
        time.sleep(5)
    print(json.dumps(report()), flush=True)
    if failed:
        raise SystemExit('Failed tasks: '+', '.join(sorted(failed)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init', 'evaluate', 'run', 'report'])
    p.add_argument('--model', choices=frozen.MODELS)
    p.add_argument('--cohort', choices=TASKS)
    p.add_argument('--task', choices=['AGE', 'SEX', 'DX', 'AD_DX'])
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if a.action == 'init':
        print(len(register()['jobs']))
    elif a.action == 'evaluate':
        evaluate(a.model, a.cohort, a.task)
    elif a.action == 'report':
        print(report())
    else:
        if a.workers < 1:
            p.error('workers must be positive')
        run(a.workers)


if __name__ == '__main__':
    main()
