#!/usr/bin/env python3
"""Train-partition-only readout selection; frozen cached encoders, no GPU."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import run_connectivity_readouts as base
import run_readout_completion as refs
from fm_matrix.data import save, sha
from benchmark_evaluation import GRID, _predict, score
from paper_paired_stats import paired, holm

RUN = ROOT/'runs/nested_readout_selection_v1'
OUT = ROOT/'results/generated/nested_readout_selection_v1'


def read(p): return json.loads(Path(p).read_text())
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def register():
    sources = {}
    for p in (Path(__file__), Path(base.__file__), ROOT/'src/benchmark_evaluation.py', ROOT/'src/paper_paired_stats.py'):
        sources[str(p)] = sha(p)
    for model in base.MODELS:
        for co in ('adni', 'nacc'):
            p = base.RUN/'features'/model/co/'complete.json'; sources[str(p)] = sha(p)
            for ep in ('MEM', 'EF'):
                p = base.RUN/'cognition'/model/f'{co}_{ep}.json'; sources[str(p)] = sha(p)
    d = dict(models=base.MODELS, cohorts=['adni','nacc'], outcomes=['MEM','EF'],
        candidates={m:base.readout_names(m) for m in base.MODELS},
        selection='All supported final normalized encoder readouts, including primary; no historical layer3/raw-layer4 arms. Per outer fold: select readout and ridge strength jointly by inner-validation Spearman correlation. Exact ties use registered readout order (primary first), then original GRID order. No outer-test outcomes used in selection.',
        evaluation='Reuse the original15 participant-grouped outer/inner splits (5 folds x3 seeds), samples and targets. StandardScaler and exact PCA<=100 fitted within inner training for selection, then refitted in full outer training for the chosen readout. All outer-test predictions are generated before paired inference. Average OOF predictions over split seeds.',
        cache_audit='Recompute selected outer-fold predictions and check against the archived same-readout OOF predictions. Default readout remains a fixed reference. Inner validation scores and selected readout/alpha stored for every fold.',
        references='Primary readout, atlas FC, atlas FC-Tucker, Schaefer100 FC, and actual-input-window FC where separately available. Preserve existing protocols and exact sample/split audit.',
        inference='5000 participant-cluster bootstraps and20000 two-sided fixed-prediction swaps. Holm within outcome x reference across all9 configurations and both cohorts, released only after36 tasks complete. No new absolute-null claims; earlier primary-readout absolute P values cannot be reused for the selected pipeline.',
        scope='Retrospective pipeline sensitivity in previously analyzed cohorts; not untouched external validation or a universal winning readout.',
        GPU_required=False, automatic_shutdown=False, source_sha256=sources)
    p = RUN/'protocol.json'
    from fm_matrix.data import clean
    if p.exists(): assert read(p) == clean(d)
    else: save(p, d)


def choose_grid(ai, bi, ytrain, yval):
    results = []
    for c in GRID:
        pred = _predict(ai, ytrain, bi, c, False)
        s = score(yval, pred, False)
        results.append(dict(c=c, ridge_alpha=1/c, validation_rho=s if np.isfinite(s) else None))
    best = max(range(len(results)), key=lambda i: -np.inf if results[i]['validation_rho'] is None else results[i]['validation_rho'])
    return results[best], results


def choose_readout(rows):
    return max(rows, key=lambda x: -np.inf if x['validation_rho'] is None else x['validation_rho'])


def task(model, co, ep):
    destination = RUN/'evaluation'/model/f'{co}_{ep}.json'
    if destination.exists(): return
    import sf100_benchmark as sf
    path = base.RUN/'cognition'/model/f'{co}_{ep}.json'
    old = read(path)
    template_path = sf.RUN/'evaluation'/co/f'{ep}.json'
    template = read(template_path)
    refs.assert_aligned(old, template, True)
    m = sf.labels(co)
    ix = m.set_index('scan').index.get_indexer(old['scan_ids'])
    assert (ix >= 0).all() and m.iloc[ix].subject.astype(str).tolist() == old['subjects']
    y, groups = np.asarray(old['y']), np.asarray(old['subjects'])
    names = base.readout_names(model)
    folder = base.RUN/'features'/model/co
    stamp = read(folder/'complete.json')
    assert stamp['scan_ids'] == m.scan.tolist()
    xs = {}
    for name in names:
        p = folder/f'{name}.npy'
        assert sha(p) == stamp['features_sha256'][name]
        xs[name] = np.load(p, mmap_mode='r')
    predictions = np.full((3, len(y)), np.nan)
    folds = []
    for f in old['split_audit']:
        split_seed, fold = f['seed'], f['fold']
        tr, te, it, iv = [np.asarray(f[k], int) for k in ('train','test','inner_train','validation')]
        assert not set(groups[tr]) & set(groups[te])
        assert not set(groups[it]) & set(groups[iv])
        assert set(it) | set(iv) == set(tr)
        candidates = []
        for name in names:
            cache = RUN/'inner'/model/co/ep/f'{split_seed}_{fold}_{name}.json'
            if cache.exists():
                selected = read(cache)
            else:
                ai, bi = base.project_indices(xs[name], ix[it], ix[iv], 100, 'cpu')
                best, grid = choose_grid(ai, bi, y[it], y[iv])
                selected = dict(readout=name, **best, grid=grid)
                save(cache, selected)
            candidates.append(selected)
        chosen = choose_readout(candidates)
        name = chosen['readout']
        outer_path = RUN/'outer'/model/co/ep/f'{split_seed}_{fold}.json'
        if outer_path.exists():
            record = read(outer_path)
            assert record['selected'] == chosen
            pred = np.asarray(record['prediction'])
        else:
            a, b = base.project_indices(xs[name], ix[tr], ix[te], 100, 'cpu')
            pred = _predict(a, y[tr], b, chosen['c'], False)
            expected = np.asarray(old['predictions_by_seed'][name][split_seed])[te]
            np.testing.assert_allclose(pred, expected, rtol=2e-5, atol=2e-5,
                err_msg='Selected readout/alpha no longer reproduces the original fitted pipeline')
            record = dict(seed=split_seed, fold=fold, selected=chosen, candidates=candidates,
                test=te, prediction=pred, cache_max_abs_error=float(abs(pred-expected).max()))
            save(outer_path, record)
        predictions[split_seed, te] = pred
        folds.append(record)
        save(RUN/'progress'/f'{model}_{co}_{ep}.json', dict(updated=utc(), fold=fold,
            split_seed=split_seed, selected_readout=name, finished_folds=len(folds), total_folds=15))
        print(model, co, ep, split_seed, fold, name, flush=True)
    assert np.isfinite(predictions).all()
    selected = predictions.mean(0)
    references = {'primary':old['predictions']['native'], 'sf100_FC':template['predictions']['sf100_fc']}
    refpath = refs.reference_root(model)/refs.atlas(model)/co/f'{ep}.json'
    reference = read(refpath)
    refs.assert_aligned(old, reference, True)
    references.update(atlas_FC=reference['predictions']['FC'], atlas_FC_Tucker=reference['predictions']['FC-Tucker'])
    sources = {str(p):sha(p) for p in (path, template_path, refpath, folder/'complete.json')}
    if model not in base.MODELS[:4]:
        wp = refs.window_root(model)/'input_comparison'/model/f'{co}.json'
        w = next(r for r in read(wp)['scores'] if r['task'] == ep)
        refs.assert_aligned(old, w)
        references['input_window_FC'] = w['predictions']['FC']
        sources[str(wp)] = sha(wp)
    rows = []
    for name, pred in references.items():
        p = RUN/'stats'/model/co/ep/f'{name}.json'
        if p.exists(): row = read(p)
        else:
            row = dict(model=model, cohort=co, task=ep, reference=name, candidate='inner_selected_readout',
                family=ep+'::'+name, **paired(y,pred,selected,groups,seed=base.seed(model+co+ep+name+'nested')))
            save(p, row)
        rows.append(row)
    save(destination, dict(status='complete', model=model, cohort=co, task=ep, rows=rows,
        y=y, subjects=groups, scan_ids=old['scan_ids'], split_audit=old['split_audit'],
        prediction=selected, predictions_by_seed=predictions, folds=folds,
        selected_rho=score(y,selected,False), source_sha256=sources))


def report():
    rows, selections, coverage = [], [], []
    for model in base.MODELS:
        for co in ('adni','nacc'):
            for ep in ('MEM','EF'):
                p = RUN/'evaluation'/model/f'{co}_{ep}.json'
                coverage.append(dict(model=model,cohort=co,task=ep,status='complete' if p.exists() else 'pending'))
                if not p.exists(): continue
                d = read(p); rows.extend(d['rows'])
                for f in d['folds']:
                    selections.append(dict(model=model,cohort=co,task=ep,seed=f['seed'],fold=f['fold'],
                        **{k:v for k,v in f['selected'].items() if k!='grid'}))
    done = all(c['status']=='complete' for c in coverage)
    OUT.mkdir(parents=True,exist_ok=True)
    if rows:
        df=pd.DataFrame(rows);df['holm_p']=np.nan
        if done:
            for fam,ix in df.groupby('family').groups.items(): df.loc[ix,'holm_p']=holm(df.loc[ix,'permutation_p'])
        df.to_csv(OUT/'Comparisons.csv',index=False)
        pd.DataFrame(selections).to_csv(OUT/'Selected_readouts.csv',index=False)
        if done:
            with pd.ExcelWriter(OUT/'Nested_readout_selection.xlsx',engine='xlsxwriter') as w:
                df.to_excel(w,sheet_name='Paired tests',index=False)
                pd.DataFrame(selections).to_excel(w,sheet_name='Fold selections',index=False)
                pd.DataFrame(coverage).to_excel(w,sheet_name='Coverage',index=False)
    save(RUN/'queue_status.json',dict(status='complete' if done else 'running', updated=utc(),
        complete=sum(c['status']=='complete' for c in coverage),total=36,tasks=coverage,automatic_shutdown=False))


def run(workers):
    lock=(RUN/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for p,h in read(RUN/'protocol.json')['source_sha256'].items(): assert sha(p)==h,p
    jobs=[(m,c,e) for m in base.MODELS for c in ('nacc','adni') for e in ('MEM','EF')
        if not (RUN/'evaluation'/m/f'{c}_{e}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(task,*job):job for job in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try: f.result()
            except Exception:
                save(RUN/'errors'/('_'.join(job)+'.json'),dict(error=traceback.format_exc(),updated=utc()))
                print('FAILED',job,traceback.format_exc(),flush=True)
            report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','report']);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    if a.action=='register':register();report()
    elif a.action=='run':run(a.workers)
    else:report()
