#!/usr/bin/env python3
"""Extend the existing source-only SF100/GFM memory axes to HRV delayed recall."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import sf100_benchmark as sf
from sample_alignment import common_mask
from fm_matrix.data import save,sha
from paper_paired_stats import paired,holm
from benchmark_evaluation import score
from readout_sensitivity import seed_of

RUN=ROOT/'runs/sf100_hrv_recall_axes_v1'

def main():
    RUN.mkdir(exist_ok=True,parents=True);scores=[];contrasts=[];files={}
    def record(p):files[str(p.relative_to(ROOT))]=sha(p)
    tm=sf.labels('hrv_pre');tx=sf.features('hrv_pre',False)
    ids=np.flatnonzero(common_mask(tm,'hrv_pre')&tm.MEM_delayed_recall.notna().to_numpy())
    y=tm.iloc[ids].MEM_delayed_recall.to_numpy(float)
    assert len(ids)==57
    for co in ('adni','nacc','hrv_pre','act'):
        record(sf.label_file(co));record(sf.RUN/'matches'/(co+'.json'))
        for method in ('sf100_fc','braingfm'):record(sf.RUN/'features'/co/(method+'.npy'))
    for source in ('adni','nacc'):
        sm=sf.labels(source);sx=sf.features(source,False)
        ix=np.flatnonzero(common_mask(sm,source)&sm.MEM.notna().to_numpy())
        p=sf.RUN/'transfer'/source/'act_MEM.json';record(p);previous=sf.read(p)
        am=sf.labels('act');ax=sf.features('act',False)
        ai=am.set_index('scan').index.get_indexer(previous['scan_ids']);assert (ai>=0).all()
        pp={};audit=[]
        for method in ('sf100_fc','braingfm'):
            selected=[r for r in previous['source_splits'] if r['representation']==method]
            assert len(selected)==3
            for r in selected:
                assert set(r['train_scan_ids'])|set(r['validation_scan_ids'])==set(sm.iloc[ix].scan)
                tr=sm[sm.scan.isin(r['train_scan_ids'])].subject
                va=sm[sm.scan.isin(r['validation_scan_ids'])].subject
                assert not set(tr)&set(va)
            sc=StandardScaler().fit(sx[method][ix]);a=sc.transform(sx[method][ix])
            pc=PCA(n_components=min(100,len(ix)-1,a.shape[1]),svd_solver='full').fit(a)
            a=pc.transform(a);b=pc.transform(sc.transform(tx[method][ids]))
            check=pc.transform(sc.transform(ax[method][ai]));pred=[];check_pred=[]
            for s in selected:
                model=Ridge(alpha=1/s['selected_c']).fit(a,sm.iloc[ix].MEM.to_numpy(float))
                pred.append(model.predict(b));check_pred.append(model.predict(check))
            # Verify that refitting reproduces an existing external target exactly.
            reproduced=np.mean(check_pred,axis=0)
            error=float(np.max(np.abs(reproduced-np.asarray(previous['predictions'][method]))))
            print('Prior-prediction check',source,method,'max_abs_error',error,flush=True)
            # Float32 PCA/BLAS refits can differ by a few ulps across thread counts.
            assert np.allclose(reproduced,previous['predictions'][method],rtol=1e-5,atol=2e-5)
            pp[method]=np.mean(pred,axis=0)
            scores.append(dict(source=source,cohort='hrv_pre',task='MEM_axis_to_MEM_delayed_recall',model=method,n=len(ids),score=score(y,pp[method],False),metric='rho'))
            audit.extend([dict(s,prior_prediction_max_abs_error=error) for s in selected])
        stat=dict(source=source,cohort='hrv_pre',task='MEM_axis_to_MEM_delayed_recall',reference='sf100_fc',candidate='braingfm',family='HRV_recall_source_axes',
                  **paired(y,pp['sf100_fc'],pp['braingfm'],tm.iloc[ids].subject,seed=seed_of(source+'hrv_recall_axes')))
        contrasts.append(stat)
        save(RUN/(source+'.json'),dict(status='complete',source=source,scan_ids=tm.iloc[ids].scan.tolist(),subjects=tm.iloc[ids].subject.tolist(),y=y,predictions=pp,
            source_scan_ids=sm.iloc[ix].scan.tolist(),source_subjects=sm.iloc[ix].subject.tolist(),source_splits=audit,
            protocol='Existing source-only grouped 80:20 tuning (seeds 0/1/2), source-only full refit; matched prior ACT predictions checked. HRV Week-2 MRI predicts Week-5 single delayed recall. No target labels used in fitting or selection.'))
        print(source,[(r['model'],r['score']) for r in scores if r['source']==source],flush=True)
    for r,p in zip(contrasts,holm([r['permutation_p'] for r in contrasts])):r.update(holm_p=p,family_size=2)
    pd.DataFrame(scores).to_csv(RUN/'scores.csv',index=False)
    pd.DataFrame(contrasts).to_csv(RUN/'paired_comparisons.csv',index=False)
    record(Path(__file__))
    save(RUN/'report.json',dict(status='complete',scores=scores,contrasts=contrasts,files_sha256=files,
        correction='Separate post hoc two-comparison Holm family; existing transfer families unchanged.',gpu_required=False))

if __name__=='__main__':main()
