#!/usr/bin/env python3
"""Additive CPU-only PCA sensitivity on cached external-adaptation embeddings.

No parent result, encoder, manuscript, or original correction family is changed.
Run `prepare`, then `run`; restart `run` to resume completed artifacts safely.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import reviewer_completion_tasks as r
import unified_supplements as u
import intervention_prepost as ip
from fm_matrix.data import save, sha
from fm_readout_v4.data import metadata
from fm_readout_v4.evaluate import tag
from intervention_prepost_inference import bootstrap_metrics, swaps
from paper_paired_stats import holm

RUN = ROOT/'runs/posttraining_pca_sensitivity_v1'
OUT = ROOT/'results/generated/posttraining_pca_sensitivity_v1'
MODELS = (*u.CONFIGS, 'brainharmonix_f')
COHORTS = ('act', 'cogte', 'breathe', 'psopt', 'hrv')
ROW_KEYS = ('task', 'mode', 'setting')


def read(p):
    return json.loads(Path(p).read_text())


def states(model):
    return list(u.states(model))


def feature_folder(state):
    if state[1] == 'brainharmonix_f':
        return ROOT/'runs/brainharmonix_f_v1/features'/tag(*state)
    return u.feature_folder(state)


def frozen_path(model, co, inference=False):
    base = (ROOT/'runs/brainharmonix_f_v1/supplements' if model == 'brainharmonix_f'
            else ROOT/'runs/reviewer_completion_v1')
    return base/('pca_inference' if inference else 'pca')/model/(co+'.json')


def fit_path(state, co):
    return RUN/'fits'/tag(*state)/'pca'/state[1]/(co+'.json')


def complete(p):
    return p.exists() and read(p).get('status') == 'complete'


def validate_rows(rows, co):
    plan = read(ip.RUN/'inputs.json')['plans'][co]
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ROW_KEYS)
        if key in seen:
            raise ValueError('Duplicate prediction row')
        seen.add(key)
        task = plan['tasks'][row['task']]
        for name in ('subjects', 'y', 'classes'):
            if row[name] != task[name]:
                raise ValueError('Changed participants/outcomes/classes: '+name)
        for p in row['predictions'].values():
            p = np.asarray(p)
            if len(p) != len(task['y']) or not np.isfinite(p).all():
                raise ValueError('Invalid predictions')
            if task['classes']:
                np.testing.assert_allclose(p.sum(1), 1, atol=1e-8)
    expected = {(task, mode, setting) for task, t in plan['tasks'].items()
                for mode in ('baseline', 'change', 'baseline_plus_change')
                for setting in (('imaging_only',) if t['classes'] else
                                ('imaging_only', 'clinical_plus_imaging'))}
    if seen != expected:
        raise ValueError('Incomplete task/input coverage')


def prepare():
    files = {}
    def record(p):
        p = Path(p)
        files[str(p)] = sha(p)
    record(ip.RUN/'inputs.json')
    plan = read(ip.RUN/'inputs.json')['plans']
    for co in COHORTS:
        for task in plan[co]['tasks'].values():
            g = np.asarray(task['subjects'])
            if len(np.unique(g)) != len(g):
                raise ValueError('Paired-visit task must have one row per participant')
            counts = np.zeros((3, len(g)), int)
            for f in task['splits']:
                tr, te, it, iv = [set(f[k]) for k in ('train', 'test', 'inner_train', 'validation')]
                if tr & te or it & iv or it | iv != tr:
                    raise ValueError('Invalid nested partition')
                counts[f['seed'], list(te)] += 1
            if not np.all(counts == 1):
                raise ValueError('OOF coverage differs from once per split seed')
    jobs = []
    for model in MODELS:
        for state in states(model):
            folder = feature_folder(state)
            stamp = read(folder/'complete.json')
            if stamp.get('status') != 'complete':
                raise ValueError('Incomplete encoder export')
            record(folder/'complete.json')
            for co in COHORTS:
                for ds in {plan[co]['pre_cohort'], plan[co]['post_cohort']}:
                    p = folder/(ds+'.npy')
                    if str(p) not in files:
                        record(p)
                        if model in u.CONFIGS[:4]:
                            # Existing audit checks these against their registered parent.
                            u.pc.verify_input_files([p])
                        elif stamp['feature_sha256'][ds] != files[str(p)]:
                            raise ValueError('Embedding hash mismatch')
                    a = np.load(p, mmap_mode='r')
                    m = metadata(ds)
                    if len(a) != len(m) or not np.isfinite(a).all():
                        raise ValueError('Embedding metadata/finite check failed')
                    for task in plan[co]['tasks'].values():
                        for visit in ('pre', 'post'):
                            if ds != plan[co][visit+'_cohort']:
                                continue
                            ix = task[visit+'_indices']
                            if m.iloc[ix].scan.tolist() != task[visit+'_scans']:
                                raise ValueError('Scan order changed')
                            if m.iloc[ix].subject.astype(str).tolist() != list(map(str, task['subjects'])):
                                raise ValueError('Participant order changed')
                if state[2] == 'frozen':
                    for p in (frozen_path(model, co), frozen_path(model, co, True)):
                        if not complete(p):
                            raise ValueError('Missing frozen sensitivity '+str(p))
                        record(p)
                    validate_rows(read(frozen_path(model, co))['rows'], co)
                jobs.append(dict(kind='fit', state=state, cohort=co,
                                 output=str(fit_path(state, co))))
        for co in COHORTS:
            jobs.append(dict(kind='inference', model=model, cohort=co,
                             output=str(RUN/'inference'/model/(co+'.json'))))
    for model in ('sf100_fc', 'braingfm'):
        for co in COHORTS:
            for inference in (False, True):
                p = frozen_path(model, co, inference)
                if not complete(p):
                    raise ValueError('Missing comparator sensitivity')
                record(p)
    codes = [Path(__file__), Path(r.__file__), Path(ip.__file__),
             ROOT/'scripts/intervention_prepost_inference.py', ROOT/'src/paper_paired_stats.py',
             ROOT/'src/benchmark_evaluation.py', ROOT/'src/benchmark_completion_tasks.py']
    protocol = dict(models=list(MODELS), cohorts=list(COHORTS), pca_grid=[5, 10, 20],
        split_seeds=[0, 1, 2], adaptation_seeds=[42, 43, 44],
        objectives=['MEM', 'EF', 'native_ssl'], sources=['adni', 'nacc'],
        tasks=['MEM_change', 'EF_change', 'arm'], HRV_memory_change='unavailable; not fabricated',
        scope='Paired-visit sensitivity only; no extension to baseline follow-up or cross-cohort axes.',
        selection='Reuse frozen PCA/ridge-logistic joint inner selection verbatim. Fixed5 fitted alongside. Same outer tests, clinical variables and inputs.',
        aggregation='Mean OOF predictions over split seeds, then over all three adaptation seeds; never average fitted embeddings.',
        contrasts=['selected_vs_fixed5', 'adapted_vs_frozen_with_both_selected'],
        inference='5000 participant bootstraps, 20000 two-sided conditional prediction swaps; no label-null refitting. Probability vectors swapped together.',
        families='contrast x task x setting x objective. Pool nine configurations, both sources, all cohorts and input modes. Frozen PCA families additionally include SF100/BrainGFM. Nonestimable planned contrasts remain p=1.',
        selection_warning='Retrospective bounded sensitivity, not prospective protocol. No best-seed or outer-test model selection.',
        manuscript_editing=False, automatic_shutdown=False, resource='CPU only')
    manifest = dict(status='registered', jobs=jobs, protocol=protocol, files_sha256=files,
                    code_sha256={str(p):sha(p) for p in codes})
    dest = RUN/'manifest.json'
    if dest.exists() and read(dest) != manifest:
        raise ValueError('Immutable manifest differs; use a new run version')
    save(dest, manifest)
    print(json.dumps(dict(jobs=len(jobs), fit_jobs=sum(j['kind']=='fit' for j in jobs),
                         reused_frozen=45, input_files=len(files))), flush=True)


def verify_manifest():
    manifest = read(RUN/'manifest.json')
    for p, h in {**manifest['files_sha256'], **manifest['code_sha256']}.items():
        if sha(p) != h:
            raise ValueError('Registered dependency changed: '+p)
    return manifest


def cached_projector():
    # Keep a strong reference to each x so Python cannot reuse an id mid-job.
    arrays, cache = {}, {}
    def project(x, tr, te, k):
        arrays[id(x)] = x
        key = (id(x), tuple(tr), tuple(te), k)
        if key not in cache:
            with patch.object(ip, 'K', k):
                cache[key] = ip.project(x, tr, te)
        return cache[key]
    return project


def fit(state, co):
    state = tuple(state)
    dest = fit_path(state, co)
    if complete(dest):
        return
    if state[2] == 'frozen':
        p = frozen_path(state[1], co)
        d = read(p)
        validate_rows(d['rows'], co)
        save(dest, {**d, 'state':state, 'reused_from':str(p), 'reused_sha256':sha(p)})
        return
    manifest = read(RUN/'manifest.json')
    def feature(s, ds):
        if tuple(s) != state:
            raise ValueError('Wrong encoder state')
        p = feature_folder(state)/(ds+'.npy')
        if sha(p) != manifest['files_sha256'][str(p)]:
            raise ValueError('Embedding changed')
        return np.load(p, mmap_mode='r')
    with patch.object(r, 'RUN', RUN/'fits'/tag(*state)), \
         patch.object(r, 'state', lambda _:state), \
         patch.object(r.core, 'feature', feature), \
         patch.object(r, 'project_k', cached_projector()):
        r.pca(state[1], co)
    d = read(dest)
    validate_rows(d['rows'], co)
    save(dest, {**d, 'state':state, 'scope':'External post-trained encoder; fixed5 versus inner-selected PCA. No encoder refit.',
                'manifest_sha256':sha(RUN/'manifest.json')})


def row_index(d):
    return {tuple(row[k] for k in ROW_KEYS):row for row in d['rows']}


def ensemble(records):
    if len(records) != 3:
        raise ValueError('Exactly three adaptation seeds required')
    a = records[0]
    for b in records[1:]:
        for key in (*ROW_KEYS, 'subjects', 'y', 'classes'):
            if b[key] != a[key]:
                raise ValueError('Unpaired seed ensemble')
    return {**a, 'predictions':{k:np.mean([r['predictions'][k] for r in records], axis=0)
                               for k in ('fixed5', 'inner_selected')}}


def contrast(row, a, b, info):
    y = np.asarray(row['y']); a = np.asarray(a); b = np.asarray(b); classes = row['classes']
    key = json.dumps(info, sort_keys=True); seed = r.seed(key)
    metric = 'auc' if classes else 'rho'
    base = dict(info, n=len(y), metric=metric, reference_score=ip.primary(y,a,classes),
                candidate_score=ip.primary(y,b,classes), bootstrap_requested=5000,
                permutations_requested=20000)
    base['delta'] = base['candidate_score'] - base['reference_score']
    if not np.isfinite(base['delta']):
        return dict(base, estimable=False, permutation_p=None, ci95=None, bootstrap_valid=0)
    stat = swaps(y, a, b, classes, seed)
    if np.array_equal(a,b):
        return dict(base, **stat, ci95=[0.,0.], bootstrap_valid=5000)
    rng = np.random.default_rng(seed); draws=[]
    for start in range(0,5000,128):
        w = rng.multinomial(len(y), np.full(len(y),1/len(y)), size=min(128,5000-start))
        draws.extend(bootstrap_metrics(y,b,w,classes)[metric]-bootstrap_metrics(y,a,w,classes)[metric])
    draws = np.asarray(draws); draws = draws[np.isfinite(draws)]
    return dict(base, **stat, ci95=np.quantile(draws,[.025,.975]) if len(draws) else None,
                bootstrap_valid=len(draws))


def inference(model, co):
    dest = RUN/'inference'/model/(co+'.json')
    if complete(dest):
        return
    frozen = row_index(read(fit_path(('none',model,'frozen','none',0),co)))
    rows=[]; summaries=[]; predictions=[]
    for source in ('adni','nacc'):
        for obj,target in [('supervised','MEM'),('supervised','EF'),('native_ssl','none')]:
            seeds = [row_index(read(fit_path((source,model,obj,target,s),co))) for s in (42,43,44)]
            if any(set(d)!=set(frozen) for d in seeds):
                raise ValueError('Frozen/adapted coverage mismatch')
            objective = target if obj=='supervised' else 'SSL'
            for key, f in frozen.items():
                row = ensemble([d[key] for d in seeds])
                for field in ('subjects','y','classes'):
                    if row[field]!=f[field]:
                        raise ValueError('Frozen/adapted participant mismatch')
                info=dict(model=model,cohort=co,source=source,objective=objective,
                          **{k:row[k] for k in ROW_KEYS})
                fam = '|'.join(str(info[k]) for k in ('task','setting','objective'))
                selected=row['predictions']['inner_selected'];fixed=row['predictions']['fixed5']
                fs=np.asarray(f['predictions']['inner_selected']);ff=np.asarray(f['predictions']['fixed5'])
                summaries.append(dict(info,n=len(row['y']),adapted_fixed5=ip.primary(row['y'],fixed,row['classes']),
                    adapted_selected=ip.primary(row['y'],selected,row['classes']),
                    frozen_fixed5=ip.primary(f['y'],ff,f['classes']),frozen_selected=ip.primary(f['y'],fs,f['classes'])))
                predictions.append(dict(info,subjects=row['subjects'],y=row['y'],classes=row['classes'],
                                        adapted_fixed5=fixed,adapted_selected=selected,frozen_selected=fs))
                for name,a,b in [('selected_vs_fixed5',fixed,selected),('adapted_vs_frozen_with_both_selected',fs,selected)]:
                    rows.append(contrast(row,a,b,dict(info,contrast=name,family=name+'|'+fam)))
    save(dest,dict(status='complete',rows=rows,summaries=summaries,predictions=predictions,
                   manifest_sha256=sha(RUN/'manifest.json')))


def execute(job):
    start=time.monotonic()
    if job['kind']=='fit':fit(job['state'],job['cohort'])
    else:inference(job['model'],job['cohort'])
    if not complete(Path(job['output'])):
        raise RuntimeError('Missing complete artifact')
    elapsed=time.monotonic()-start
    save(Path(job['output']).with_suffix('.timing.json'),dict(status='complete',seconds=elapsed))
    return elapsed


def export():
    manifest=verify_manifest()
    if not all(complete(Path(j['output'])) for j in manifest['jobs']):
        raise ValueError('Cannot export incomplete experiment')
    rows=[];summaries=[];choices=[]
    for model in (*MODELS,'sf100_fc','braingfm'):
        for co in COHORTS:
            for q in read(frozen_path(model,co,True))['rows']:
                fam='selected_vs_fixed5|'+q['task']+'|'+q['setting']+'|frozen'
                rows.append({**q,'source':'none','objective':'frozen','contrast':'selected_vs_fixed5','family':fam})
            for row in read(frozen_path(model,co))['rows']:
                summaries.append(dict(model=model,cohort=co,source='none',objective='frozen',
                    **{k:row[k] for k in ROW_KEYS},n=len(row['y']),
                    frozen_fixed5=row['scores']['fixed5'],frozen_selected=row['scores']['inner_selected']))
            if model not in MODELS:continue
            d=read(RUN/'inference'/model/(co+'.json'));rows+=d['rows'];summaries+=d['summaries']
            for s in states(model):
                for row in read(fit_path(s,co))['rows']:
                    for choice in row['selected']:
                        choices.append(dict(model=model,cohort=co,source=s[0],objective=s[2],target=s[3],
                            adaptation_seed=s[4],**{k:row[k] for k in ROW_KEYS},**choice))
    families=defaultdict(list)
    for i,row in enumerate(rows):families[row['family']].append(i)
    for ids in families.values():
        for i,p in zip(ids,holm([rows[i].get('permutation_p') for i in ids])):
            rows[i]['holm_p']=float(p);rows[i]['family_size']=len(ids)
    OUT.mkdir(parents=True,exist_ok=True)
    frames={'Scores':pd.DataFrame(summaries),'Paired_comparisons':pd.DataFrame(rows),'PC_selections':pd.DataFrame(choices)}
    for name,df in frames.items():df.to_csv(OUT/(name+'.csv'),index=False)
    pd.DataFrame([{'item':k,'definition':str(v)} for k,v in manifest['protocol'].items()]).to_csv(OUT/'Protocol.csv',index=False)
    save(RUN/'complete.json',dict(status='complete',manifest_sha256=sha(RUN/'manifest.json'),
        comparisons=len(rows),families=len(families),scores=len(summaries),
        files_sha256={str(p):sha(p) for p in OUT.iterdir() if p.is_file()}))


def run(workers):
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=verify_manifest(); jobs=manifest['jobs']; pending={};failed=[];durations=[]
    started=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            done=sum(complete(Path(j['output'])) for j in jobs)
            for job in jobs:
                if len(pending)>=workers:break
                if complete(Path(job['output'])) or job in pending.values() or job in failed:continue
                if job['kind']=='inference' and not all(complete(fit_path(s,job['cohort'])) for s in states(job['model'])):continue
                pending[pool.submit(execute,job)]=job
            save(RUN/'status.json',dict(status='running' if pending else 'failed' if failed else 'complete',
                total=len(jobs),complete=done,failed=len(failed),workers=workers,pid=os.getpid(),
                started_at=started,updated_at=time.time(),running=list(pending.values()),
                mean_recent_job_seconds=float(np.mean(durations[-20:])) if durations else None))
            if not pending:break
            finished,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for f in finished:
                job=pending.pop(f)
                try:
                    seconds=f.result();durations.append(seconds)
                    print('complete',job['kind'],job.get('state',job.get('model')),job['cohort'],round(seconds,2),flush=True)
                except Exception:
                    failed.append(job)
                    save(Path(job['output']).with_suffix('.error.json'),dict(status='failed',job=job,error=traceback.format_exc()))
                    print('FAILED',job,traceback.format_exc(),flush=True)
    if failed:raise RuntimeError(f'{len(failed)} jobs failed; inspect *.error.json and resume')
    export()
    save(RUN/'status.json',dict(status='complete',total=len(jobs),complete=len(jobs),failed=0,
                              updated_at=time.time(),elapsed_seconds=time.time()-started,report=str(OUT)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','export','fit']);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--state',nargs=5);p.add_argument('--cohort',choices=COHORTS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a.workers)
    elif a.action=='export':export()
    else:
        s=a.state;s[-1]=int(s[-1]);fit(tuple(s),a.cohort)
