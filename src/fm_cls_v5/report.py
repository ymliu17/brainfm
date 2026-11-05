"""Descriptive results with explicit seed and participant alignment."""
import json
import numpy as np
import pandas as pd
from fm_cls_v5.data import ROOT,RUN,save
from fm_cls_v5.train import location
from fm_cls_v5.evaluate import tag
from benchmark_evaluation import score


def report():
    cv=[];external=[];summaries=[]
    for source in ('adni','nacc'):
        for target in ('MEM','EF'):
            predictions=[]
            for seed in (42,43,44):
                path=location(source,'brainlm13','supervised',target,seed,'cv')/'complete.json'
                d=json.loads(path.read_text());folds=d['folds']
                frame=pd.DataFrame({key:np.concatenate([f[key] for f in folds]) for key in
                    ('test_indices','test_subjects','y','frozen_prediction','adapted_prediction')}).sort_values('test_indices').reset_index(drop=True)
                if frame.test_indices.duplicated().any():raise ValueError('Duplicate CV test scan')
                if len(folds)!=5 or any(set(f['test_subjects'])&(set(f['train_subjects'])|set(f['validation_subjects'])) for f in folds):raise ValueError('Invalid participant CV')
                if predictions:
                    ref=predictions[0]
                    if not frame[['test_indices','test_subjects']].equals(ref[['test_indices','test_subjects']]) or not np.allclose(frame.y,ref.y):raise ValueError('Unmatched seed-level CV rows')
                predictions.append(frame)
                cv.append(dict(source=source,target=target,seed=seed,n=len(frame),n_subjects=frame.test_subjects.nunique(),
                    frozen_rho=score(frame.y,frame.frozen_prediction,False),adapted_rho=score(frame.y,frame.adapted_prediction,False),
                    selected_epochs=[f['best_epoch'] for f in folds],file=str(path)))
            pooled=predictions[0].copy()
            for key in ('frozen_prediction','adapted_prediction'):pooled[key]=np.mean([f[key].to_numpy() for f in predictions],axis=0)
            pooled.to_csv(RUN/f'{source}_{target}_seed_averaged_oof.csv',index=False)
            rows=[r for r in cv if r['source']==source and r['target']==target]
            frozen=score(pooled.y,pooled.frozen_prediction,False);adapted=score(pooled.y,pooled.adapted_prediction,False)
            summaries.append(dict(source=source,target=target,n=len(pooled),n_subjects=pooled.test_subjects.nunique(),
                mean_seed_frozen_rho=np.mean([r['frozen_rho'] for r in rows]),mean_seed_adapted_rho=np.mean([r['adapted_rho'] for r in rows]),
                seed_averaged_oof_frozen_rho=frozen,seed_averaged_oof_adapted_rho=adapted,seed_averaged_oof_delta_rho=adapted-frozen))
            for seed in (42,43,44):
                name=tag(source,'brainlm13','supervised',target,seed)
                for path in sorted((RUN/'evaluation'/name).glob('*_local.json')):
                    d=json.loads(path.read_text());reference=ROOT/'runs/fm_readout_v4/evaluation/brainlm13_frozen'/path.name
                    ref=json.loads(reference.read_text());lookup={(r['cohort'],r['task'],r['method']):r for r in ref['rows']}
                    for row in d['rows']:
                        r=lookup.get((row['cohort'],row['task'],row['method']))
                        if r is None or r['subjects']!=row['subjects'] or not np.allclose(r['y'],row['y']):raise ValueError('Unmatched frozen/adapted external rows')
                        external.append(dict(source=source,target=target,seed=seed,cohort=row['cohort'],task=row['task'],method=row['method'],
                            n=row['n'],metric=row['metric'],frozen_score=r['score'],adapted_score=row['score'],
                            delta_score=None if r['score'] is None or row['score'] is None else row['score']-r['score'],file=str(path)))
    pd.DataFrame(cv).to_csv(RUN/'within_source_by_seed.csv',index=False)
    pd.DataFrame(summaries).to_csv(RUN/'within_source_summary.csv',index=False)
    pd.DataFrame(external).to_csv(RUN/'external_local_comparisons.csv',index=False)
    save(RUN/'report.json',dict(status='complete',within_source=summaries,within_source_by_seed=cv,external_local_comparisons=external,
        interpretation='Descriptive results; no new p values or significance claims. Mean of seed scores and score of seed-averaged OOF predictions are reported separately.',
        scope='MEM/EF final-CLS training and evaluation. Source-CV uses independent fold encoders; external probes use source-only checkpoints. LOSO, calibration, axes and identification retain separate detailed outputs.',
        physiology='Original physiological results retained; no claim that the new CLS-trained encoder was evaluated on time-varying physiology.'))
