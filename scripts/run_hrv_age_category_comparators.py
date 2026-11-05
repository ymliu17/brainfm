#!/usr/bin/env python3
"""Priority CPU supplement: HRV-ER age categories for FC and BrainGFM.

Reuses the nine-FM demographic protocol, labels, participants and probe seeds.
Produces descriptive multiclass results; no new permutation test or figure edit.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import demographic_benchmark as demo
import run_evidence_nulls as engine
from benchmark_evaluation import prepare_nested
RUN=ROOT/'runs/hrv_age_category_comparators_v1'
OUT=ROOT/'results/generated/hrv_age_category_comparators_v1'
SF=ROOT/'runs/sf100_benchmark_v1'
read,save,sha=engine.read,engine.save,engine.sha

def fm_file(model,co):
    if model in ('brainlm13','brainlm111','brainlm650','brainjepa'):
        return ROOT/'runs/sample_alignment_v1/demographic_benchmark_v1/evaluation'/(model+'_frozen')/(co+'_AGE_GROUP.json')
    if model in ('semantoks','fmrilm'):
        base=ROOT/'runs/new_fm_v1';name=model
    elif model in ('fmrilm_mae','fmrilm_vq_domain'):
        base=ROOT/'runs/fmrilm_variants_v1'/('mae' if model=='fmrilm_mae' else 'vq-domain');name='fmrilm'
    else:base=ROOT/'runs/brainharmonix_f_v1';name=model
    return base/'evaluation'/(name+'_frozen')/co/'AGE_GROUP.json'

def fit(x,t):
    y=np.asarray(t['y']);g=np.asarray(t['subjects']);pred=[];audit=[]
    for seed in (0,1,2):
        folds=prepare_nested([x],[100],y,g,seed)
        current=dict(seed=seed,folds=[dict(train_subjects=g[tr].tolist(),test_subjects=g[te].tolist(),
            inner_train_subjects=g[it].tolist(),inner_validation_subjects=g[iv].tolist()) for tr,te,_,it,iv,_ in folds])
        assert current==t['split_audit'][seed],'Participant partition differs from the existing demographic protocol'
        audit.append(current);pred.append(demo.predict_multiclass(folds,y,len(demo.AGE_BINS)))
    p=np.mean(pred,axis=0)
    assert p.shape==(len(y),7) and np.isfinite(p).all() and np.allclose(p.sum(1),1)
    return p,pred,audit

def main():
    RUN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    jobs=[dict(id=co+'__'+method,cohort=co,method=method,priority='highest')
          for co in ('hrv_pre','hrv_post') for method in ('sf100_fc','braingfm')]
    protocol=dict(task='AGE_GROUP',classes=list(demo.AGE_BINS),primary_metric='balanced_accuracy',
        other_metrics=['accuracy','log_loss'],outer_folds=5,inner_validation_fraction=.2,split_seeds=[0,1,2],
        PCA='Training-only StandardScaler and PCA with at most 100 components, capped by training rank.',
        selection='Existing demographic multiclass logistic regression; C selected by inner balanced accuracy.',
        aggregation='Mean out-of-fold probability vectors across the three probe seeds, then score.',
        inference='Descriptive, matching existing exploratory age-category results. No new above-chance or paired superiority tests.',
        priority='Run immediately on the CPU reserved alongside the seven-worker permutation queue.',automatic_shutdown=False)
    save(RUN/'manifest.json',dict(jobs=jobs,protocol=protocol,code_sha256=sha(__file__)))
    summaries=[];done=[]
    for co in ('hrv_pre','hrv_post'):
        template_path=fm_file('brainlm13',co);t=read(template_path)
        for model in engine.MODELS:
            p=fm_file(model,co);r=read(p)
            assert r['scan_ids']==t['scan_ids'] and r['subjects']==t['subjects'] and np.array_equal(r['y'],t['y'])
            recalculated=demo.metrics('AGE_GROUP',np.asarray(r['y']),np.asarray(r['prediction']))
            for k,v in recalculated.items():assert abs(v-r['metrics'][k])<1e-10
            summaries.append(dict(model=model,cohort=co,n_subjects=len(t['y']),**recalculated,source_file=str(p)))
        # Validate the reused fitter against an archived FM probability array.
        fp=engine.feature_path('brainlm13',co);meta=read(SF/'features'/co/'complete.json')
        lookup={s:i for i,s in enumerate(meta['scan_ids'])};ix=[lookup[s] for s in t['scan_ids']]
        x=np.load(fp,mmap_mode='r')[ix];p,_,_=fit(x,t)
        err=float(np.max(abs(p-np.asarray(t['prediction']))))
        agreement=bool(np.array_equal(p.argmax(1),np.asarray(t['prediction']).argmax(1)))
        print('Protocol validation',co,'maximum probability difference',err,'same predicted classes',agreement,flush=True)
        assert err<.002 and agreement,(err,agreement)
        save(RUN/(co+'_protocol_validation.json'),dict(status='complete',max_prediction_error=err,identical_predicted_classes=agreement,numerical_note='Identical partitions and predicted categories; minor floating-point solver differences in probability vectors.',
             template_sha256=sha(template_path),feature_sha256=sha(fp),n_subjects=len(t['y'])))
        for method in ('sf100_fc','braingfm'):
            job=co+'__'+method;dest=RUN/'evaluation'/(job+'.json')
            save(RUN/'status.json',dict(status='running',running=job,complete_jobs=len(done),total_jobs=4,updated_utc=engine.utc(),automatic_shutdown=False))
            start=time.monotonic();fp=SF/'features'/co/(method+'.npy')
            assert sha(fp)==meta['files_sha256'][method]
            if dest.exists():r=read(dest);assert r['features_sha256']==sha(fp)
            else:
                x=np.load(fp,mmap_mode='r')[ix];assert np.isfinite(x).all()
                pred,by,audit=fit(x,t)
                r=dict(status='complete',model=method,cohort=co,task='AGE_GROUP',scan_ids=t['scan_ids'],subjects=t['subjects'],y=t['y'],
                       classes=list(demo.AGE_BINS),prediction=pred,predictions_by_probe_seed=by,split_audit=audit,
                       metrics=demo.metrics('AGE_GROUP',np.asarray(t['y']),pred),features_sha256=sha(fp),
                       template_sha256=sha(template_path),protocol=protocol,seconds=time.monotonic()-start)
                save(dest,r)
            summaries.append(dict(model=method,cohort=co,n_subjects=len(t['y']),**r['metrics'],source_file=str(dest)))
            done.append(job);print(job,r['metrics'],flush=True)
    df=pd.DataFrame(summaries);assert len(df)==22
    df.to_csv(OUT/'Age_category_scores.csv',index=False)
    with pd.ExcelWriter(OUT/'HRV_ER_age_category_results.xlsx',engine='xlsxwriter') as w:
        df.to_excel(w,sheet_name='Frozen_all_methods',index=False)
        pd.DataFrame([(k,json.dumps(v) if not isinstance(v,str) else v) for k,v in protocol.items()],columns=['Field','Definition']).to_excel(w,sheet_name='Protocol',index=False)
    (OUT/'README.txt').write_text('HRV-ER exploratory seven-category age classification. Baseline (61 participants) and follow-up (51) are evaluated separately. All eleven methods use matching participants and the existing demographic protocol. Scores use seed-averaged out-of-fold probabilities. No significance claims or heatmap changes are made. Original FM results are preserved.\n')
    save(RUN/'status.json',dict(status='complete',complete_jobs=4,total_jobs=4,updated_utc=engine.utc(),outputs=str(OUT),automatic_shutdown=False))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(RUN/'status.json',dict(status='failed',error=str(exc),updated_utc=engine.utc()));raise
