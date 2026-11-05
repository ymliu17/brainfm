#!/usr/bin/env python3
"""Frozen final-layer connectivity versus primary readouts on paired visits.

Reuse the existing paired-visit engine, participants, partitions and fixed5 PCA.
All outputs are additive. No encoder inference/training or shutdown actions.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
import numpy as np
import pandas as pd
import intervention_prepost as ip
import intervention_prepost_inference as stats
import run_connectivity_readouts as frozen
import run_posttraining_pca_sensitivity as sensitivity
from fm_matrix.data import save, sha
from fm_readout_v4.data import metadata
from paper_paired_stats import holm

RUN = ROOT/'runs/intervention_readouts_v1'
OUT = ROOT/'results/generated/intervention_readouts_v1'
COHORTS = ip.COHORTS
MODELS = frozen.MODELS


def read(p):
    return json.loads(Path(p).read_text())


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def feature_path(model, co):
    return frozen.RUN/'features'/model/co/'connectivity.npy'


def jobs():
    return [dict(id=model+'/'+co, model=model, cohort=co,
                 output='results/'+model+'/'+co+'.json') for model in MODELS for co in COHORTS]


def validate_row(row, plan):
    t = plan['tasks'][row['task']]
    for key in ('subjects', 'y', 'classes'):
        if list(row[key]) != list(t[key]):
            raise ValueError('Changed paired-visit '+key)
    if len(set(t['subjects'])) != len(t['subjects']):
        raise ValueError('Expected one paired record per participant')


def prepare():
    parent = read(ip.RUN/'inputs.json')
    files = {}
    def record(p):
        p = Path(p)
        if str(p) not in files:
            files[str(p)] = sha(p)
        return files[str(p)]
    record(ip.RUN/'inputs.json')
    for model in MODELS:
        for co in COHORTS:
            plan = parent['plans'][co]
            pp = sensitivity.frozen_path(model, co)
            record(pp)
            d = read(pp)
            assert d['status'] == 'complete'
            sensitivity.validate_rows(d['rows'], co)
            for visit in ('pre', 'post'):
                ds = plan[visit+'_cohort']
                folder = frozen.RUN/'features'/model/ds
                stamp = read(folder/'complete.json')
                record(folder/'complete.json')
                assert stamp['status'] == 'complete'
                assert stamp['layer'].startswith('final encoder output')
                assert stamp['scan_ids'] == metadata(ds).scan.tolist()
                p = feature_path(model, ds)
                assert record(p) == stamp['features_sha256']['connectivity']
                x = np.load(p, mmap_mode='r')
                assert len(x) == len(stamp['scan_ids']) and np.isfinite(x).all()
                for t in plan['tasks'].values():
                    ix = t[visit+'_indices']
                    assert [stamp['scan_ids'][i] for i in ix] == t[visit+'_scans']
                    assert metadata(ds).iloc[ix].subject.astype(str).tolist() == list(map(str,t['subjects']))
            sf = sensitivity.frozen_path('sf100_fc', co)
            record(sf)
            sensitivity.validate_rows(read(sf)['rows'], co)
            clinical = ip.RUN/'jobs/none_brainlm13_frozen_none_0'/(co+'.json')
            record(clinical)
    codes = [Path(__file__), Path(ip.__file__), Path(stats.__file__),
             ROOT/'src/benchmark_evaluation.py', ROOT/'src/paper_paired_stats.py']
    protocol = dict(models=list(MODELS), cohorts=list(COHORTS),
        readouts=['primary', 'final_normalized_embedding_connectivity'],
        tasks=['MEM_change', 'EF_change', 'arm'],
        unavailable='HRV-ER MEM change: no paired memory assessment; omitted, never imputed.',
        inputs=list(ip.MODES), settings=['imaging_only','clinical_plus_imaging for cognitive change'],
        fitting='Exact existing paired-visit engine: fold-fitted scaling and at most five PCs; ridge/logistic tuning on existing inner validation; five outer folds and split seeds 0,1,2.',
        primary='Reuse fixed5 predictions from completed frozen PCA sensitivity. Same primary readout, samples, partitions and engine. No outcome-based readout selection.',
        change='Float64 post-minus-pre embedding connectivity. Concatenate baseline and change before one PCA. This is not connectivity computed from a difference of token embeddings.',
        comparisons=['connectivity_vs_primary','connectivity_vs_sf100','connectivity_added_to_clinical','connectivity_input_vs_baseline'],
        inference='5000 participant bootstrap draws; 20000 two-sided swaps of fixed predictions; entire probability vector swapped for arm classification; no absolute null tests or refitting.',
        families='Comparison type x endpoint x imaging-only/clinical-plus-imaging setting, pooling nine configurations, all cohorts and relevant input modes. Separate exploratory families from existing primary results. Nonestimable contrasts retained at P=1.',
        scope='Bounded frozen-readout sensitivity; not supervised/SSL readout interaction, follow-up prognosis, LOSO or baseline cognition.',
        encoder_training=False, encoder_inference=False, automatic_shutdown=False,
        original_manuscript_modified=False)
    manifest = dict(jobs=jobs(), protocol=protocol, files_sha256=files,
                    code_sha256={str(p):sha(p) for p in codes})
    p = RUN/'manifest.json'
    if p.exists() and read(p) != manifest:
        raise ValueError('Registered protocol changed; use a new run version')
    save(p, manifest)
    return dict(jobs=len(manifest['jobs']), registered_files=len(files))


def verify_code():
    m = read(RUN/'manifest.json')
    for p,h in m['code_sha256'].items():
        if sha(p) != h:
            raise ValueError('Registered code changed: '+p)
    return m


def check_file(p, manifest):
    if sha(p) != manifest['files_sha256'][str(p)]:
        raise ValueError('Registered input changed: '+str(p))


def original_rows(path):
    return {(r['task'], r['mode']+('+clinical' if r['setting']=='clinical_plus_imaging' else '')):r
            for r in read(path)['rows']}


def compare(y, a, b, classes, info):
    metric = 'auc' if classes else 'rho'
    seed = frozen.seed(json.dumps(info, sort_keys=True))
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0,5000,128):
        w = rng.multinomial(len(y), np.full(len(y),1/len(y)), size=min(128,5000-start))
        draws.extend(stats.bootstrap_metrics(y,b,w,classes)[metric]-stats.bootstrap_metrics(y,a,w,classes)[metric])
    valid = np.asarray(draws);valid = valid[np.isfinite(valid)]
    sa,sb = ip.primary(y,a,classes),ip.primary(y,b,classes)
    if np.isfinite(sa) and np.isfinite(sb) and len(valid)<4750:
        raise ValueError('Insufficient valid bootstrap draws')
    return dict(**info, metric=metric, reference_score=sa, candidate_score=sb, delta=sb-sa,
        ci95=np.quantile(valid,[.025,.975]) if len(valid) else [None,None],
        n_subjects=len(y), bootstrap_requested=5000, bootstrap_valid=len(valid), permutations_requested=20000,
        uncertainty='Conditional on fitted, split-seed-averaged predictions; pointwise participant-bootstrap intervals.',
        **stats.swaps(y,a,b,classes,seed))


def evaluate(model, co):
    dest = RUN/'results'/model/(co+'.json')
    if dest.exists():
        return
    manifest = verify_code()
    check_file(ip.RUN/'inputs.json',manifest)
    parent = read(ip.RUN/'inputs.json');plan = parent['plans'][co]
    primary_path = sensitivity.frozen_path(model,co);sf_path = sensitivity.frozen_path('sf100_fc',co)
    clinical_path = ip.RUN/'jobs/none_brainlm13_frozen_none_0'/(co+'.json')
    for p in (primary_path,sf_path,clinical_path):check_file(p,manifest)
    primary = original_rows(primary_path);sf = original_rows(sf_path)
    clinical = {r['task']:r for r in read(clinical_path)['rows'] if r['mode']=='clinical'}
    state = ('none',model,'frozen','connectivity',0)
    fit_root = RUN/'fits'/model/co
    data = copy.deepcopy(parent);data['states']=[state];data['files_sha256']={}
    for ds in {plan['pre_cohort'],plan['post_cohort']}:
        p = feature_path(model,ds);check_file(p,manifest)
        data['files_sha256'][str(p)] = manifest['files_sha256'][str(p)]
    data['signature'] = frozen.seed(json.dumps(dict(parent=parent['signature'],model=model,files=data['files_sha256']),sort_keys=True))
    save(fit_root/'inputs.json',data)
    with patch.object(ip,'RUN',fit_root), patch.object(ip,'feature_path',lambda s,ds:feature_path(model,ds)):
        fit = Path(ip.run_job((state,co)))
    fitted = read(fit)
    rr = {(r['task'],r['mode']):r for r in fitted['rows']}
    expected = set(primary)
    if set(rr)-{(r['task'],r['mode']) for r in fitted['rows'] if r['mode'] in ('clinical','null')} != expected:
        raise ValueError('Incomplete input/setting coverage')
    contrasts=[];scores=[]
    for key in sorted(expected):
        row=rr[key];native=primary[key];ref=sf[key]
        for r in (row,native,ref):validate_row(r,plan)
        y=np.asarray(row['y']);b=np.asarray(row['prediction']);classes=row['classes']
        task,mode=key
        setting='clinical_plus_imaging' if '+clinical' in mode else 'imaging_only'
        info=dict(model=model,cohort=co,task=task,mode=mode,setting=setting)
        for name,pred in [('connectivity',b),('primary',np.asarray(native['predictions']['fixed5'])),('sf100_fc',np.asarray(ref['predictions']['fixed5']))]:
            scores.append(dict(**info,readout=name,n_subjects=len(y),**ip.metrics(y,pred,classes)))
        refs=[('connectivity_vs_primary','primary',np.asarray(native['predictions']['fixed5'])),
              ('connectivity_vs_sf100','sf100_fc',np.asarray(ref['predictions']['fixed5']))]
        if '+clinical' in mode:
            validate_row(clinical[task],plan)
            np.testing.assert_allclose(rr[(task,'clinical')]['prediction'],clinical[task]['prediction'],atol=1e-12,rtol=1e-12)
            refs.append(('connectivity_added_to_clinical','clinical',np.asarray(clinical[task]['prediction'])))
        if mode.startswith(('change','baseline_plus_change')):
            baseline='baseline'+('+clinical' if '+clinical' in mode else '')
            refs.append(('connectivity_input_vs_baseline','connectivity_baseline',np.asarray(rr[(task,baseline)]['prediction'])))
        for contrast,reference,a in refs:
            part=RUN/'parts'/model/co/(task+'_'+mode+'_'+contrast+'.json')
            if part.exists():stat=read(part)
            else:
                stat=compare(y,a,b,classes,dict(**info,contrast=contrast,reference=reference,candidate='connectivity',
                    family='|'.join([contrast,task,setting])))
                save(part,stat)
            contrasts.append(stat)
    save(dest,dict(status='complete',scores=scores,contrasts=contrasts,
        manifest_sha256=sha(RUN/'manifest.json'),fit=str(fit),fit_sha256=sha(fit),
        reused_primary=str(primary_path),reused_primary_sha256=sha(primary_path)))


def report():
    manifest=read(RUN/'manifest.json');coverage=[];scores=[];contrasts=[]
    for j in manifest['jobs']:
        p=RUN/j['output'];coverage.append(dict(id=j['id'],status='complete' if p.exists() else 'pending'))
        if p.exists():
            d=read(p);assert d['manifest_sha256']==sha(RUN/'manifest.json')
            scores.extend(d['scores']);contrasts.extend(d['contrasts'])
    done=all(r['status']=='complete' for r in coverage)
    a=pd.DataFrame(scores);b=pd.DataFrame(contrasts);cov=pd.DataFrame(coverage)
    if len(b):
        b['holm_p']=np.nan;b['inference_status']='pending_complete_panel'
        if done:
            for _,ix in b.groupby('family').groups.items():
                b.loc[ix,'holm_p']=holm(b.loc[ix,'permutation_p'].fillna(1).tolist())
            b['inference_status']='final_nine_configuration_family'
    OUT.mkdir(parents=True,exist_ok=True)
    for name,d in [('Scores',a),('Comparisons',b),('Coverage',cov)]:d.to_csv(OUT/(name+'.csv'),index=False)
    tmp=OUT/'Intervention_readouts.tmp.xlsx'
    with pd.ExcelWriter(tmp,engine='xlsxwriter') as w:
        for name,d in [('Scores',a),('Comparisons',b),('Coverage',cov)]:d.to_excel(w,sheet_name=name,index=False)
        pd.DataFrame([dict(item=k,description=str(v)) for k,v in manifest['protocol'].items()]).to_excel(w,sheet_name='Protocol',index=False)
    tmp.replace(OUT/'Intervention_readouts.xlsx')
    result=dict(total=len(coverage),completed=sum(r['status']=='complete' for r in coverage),all_complete=done,
        comparisons=len(b),automatic_shutdown=False,updated_utc=utc())
    save(RUN/'summary.json',result)
    if done:save(RUN/'complete.json',result)
    return result


def run(workers):
    manifest=verify_code();lock=open(RUN/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=ROOT/'logs/intervention_readouts_v1';logs.mkdir(parents=True,exist_ok=True)
    # Small cohorts first for rapid end-to-end completion; all models remain registered.
    ordered=sorted(manifest['jobs'],key=lambda j:(COHORTS.index(j['cohort']) if j['cohort']!='psopt' else -1,j['model']!='semantoks',j['model']))
    active={};failed=set();last_report=0
    while True:
        for key,(proc,fh) in list(active.items()):
            if proc.poll() is None:continue
            fh.close();del active[key]
            if proc.returncode:failed.add(key)
            print('finished',key,proc.returncode,flush=True)
        done={j['id'] for j in ordered if (RUN/j['output']).exists()}
        for j in ordered:
            if len(active)>=workers:break
            if j['id'] in done|set(active)|failed:continue
            fh=open(logs/(j['id'].replace('/','_')+'.log'),'a')
            cmd=[sys.executable,str(Path(__file__).resolve()),'evaluate','--model',j['model'],'--cohort',j['cohort']]
            active[j['id']]=(subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT),fh)
            print('started',j['id'],flush=True)
        save(RUN/'status.json',dict(total=len(ordered),completed=len(done),running=list(active),failed=sorted(failed),
            pending=len(ordered)-len(done)-len(active)-len(failed),workers=workers,automatic_shutdown=False,updated_utc=utc()))
        if time.time()-last_report>180:report();last_report=time.time()
        if len(done)==len(ordered) or (not active and failed):break
        time.sleep(5)
    print(report(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','evaluate','run','report'])
    p.add_argument('--workers',type=int,default=6);p.add_argument('--model',choices=MODELS);p.add_argument('--cohort',choices=COHORTS)
    a=p.parse_args()
    if a.action=='prepare':print(prepare())
    elif a.action=='evaluate':evaluate(a.model,a.cohort)
    elif a.action=='run':run(a.workers)
    else:print(report())


if __name__=='__main__':main()
