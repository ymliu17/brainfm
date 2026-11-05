"""Bounded post hoc robustness panel; no encoder fitting or shutdown actions."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from benchmark_evaluation import grouped_splits, predict_prepared, score
from fm_matrix.data import ROOT, WORK, save, sha
from fm_readout_v4.data import metadata, dataset, endpoints, DATA
from paper_completion import feature_dir, primary_mask, project_block, MODELS, TRIALS
from paper_paired_stats import paired, holm, weighted_scores

RUN=ROOT/'runs/paper_robustness_v1'
COHORTS=('adni','nacc',*TRIALS)
SHORT=80


def read(p):return json.loads(Path(p).read_text())
def atlas_for(model):return 'sch450' if model=='brainjepa' else 'a424'
def digest_seed(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)
def label_path(co):return DATA/(co+'_domains.csv' if co.startswith('hrv') else co+'.csv')
def cache_path(atlas,co):return ROOT/'runs/fm_posttraining_matrix_v2/baselines'/atlas/(co+'_FC.npy')


def merge_columns(m, table, key, columns):
    if table[key].duplicated().any():raise ValueError('Ambiguous metadata join: '+key)
    t=table.set_index(key)
    lookup=m['scan'] if key=='FILE_NAME' else m['subject']
    for dest,source in columns.items():m[dest]=lookup.map(t[source])


def assemble_labels(co):
    m=metadata(co).copy();sources=[label_path(co)]
    dem=ROOT/'runs/demographic_benchmark_v1/labels'/(co+'.csv')
    d=pd.read_csv(dem).set_index('scan');sources.append(dem)
    if d.index.has_duplicates:raise ValueError('Duplicate demographic scan')
    m['sex']=m.scan.map(d.sex)
    if co=='adni':
        ps=[WORK/f'data/derived/adni424_t200/ADNI_manifest_raw_ADNI_{t}_all_t200_off.csv' for t in ('MEM','EF')]
        t=pd.concat([pd.read_csv(p) for p in ps]);sources+=ps
        for key,part in t.groupby('FILE_NAME'):
            for c in ('native_tr','SITE','PTEDUCAT'):
                if part[c].dropna().nunique()>1:raise ValueError('Conflicting ADNI covariate '+key+' '+c)
        t=t.drop_duplicates('FILE_NAME')
        merge_columns(m,t,'FILE_NAME',dict(native_tr='native_tr',site='SITE',education='PTEDUCAT'))
    elif co=='nacc':
        p=WORK/'data/derived/benchmark_extension_v1/nacc_default.csv';t=pd.read_csv(p);sources.append(p)
        merge_columns(m,t,'subject',dict(native_tr='native_tr',site='SITE',education='education_years'))
        selected=m.subject.map(t.set_index('subject').session.astype(str)).str.replace('ses-','',regex=False)
        if not selected.eq(m.visit.astype(str)).all():raise ValueError('NACC clinical visit mismatch')
    elif co=='act':
        p=ROOT/'data/derived/rct/act_covariates.csv';sources.append(p)
        merge_columns(m,pd.read_csv(p),'ID',dict(site='study_site',arm='group_assignment',education='formj_subj_education_years'))
    elif co in ('cogte','breathe','psopt'):
        p=ROOT/f'data/rct/{co}/labels.csv';sources.append(p)
        cols={'arm':{'cogte':'GROUP_1.0','breathe':'group_assignment_T1','psopt':'arm_name'}[co]}
        if co=='psopt':cols.update(mean_fd='mean_fd_T1',education='education_bsl')
        merge_columns(m,pd.read_csv(p),'ID',cols)
    if co in TRIALS[:-1]:
        from temporal import COHORT_TR
        m['native_tr']=COHORT_TR[co]
    if co=='hrv_pre':
        p=WORK/'data/derived/external_preliminary_v1/hrv_pre.csv';t=pd.read_csv(p);sources.append(p)
        merge_columns(m,t,'subject',dict(native_tr='native_tr'))
    for c in ('site','arm','age_category'):
        if c in m:m[c]=m[c].astype('string')
    for c in ('age','sex','education','native_tr','native_frames','mean_fd'):
        if c in m:m[c]=pd.to_numeric(m[c],errors='raise').replace([np.inf,-np.inf],np.nan)
    return m,sources


def audit():
    files={};cohorts={};sidecars=[]
    for co in COHORTS:
        m,paths=assemble_labels(co)
        if m.scan.duplicated().any():raise ValueError('Duplicate scan')
        out=RUN/'labels'/(co+'.csv');out.parent.mkdir(parents=True,exist_ok=True);m.to_csv(out,index=False)
        paths.append(out)
        for model in MODELS:
            folder=feature_dir(('none',model,'frozen','none',0));p=folder/(co+'.npy')
            f=np.load(p,mmap_mode='r')
            if len(f)!=len(m) or not np.isfinite(f).all():raise ValueError('Invalid frozen feature')
            paths += [p,folder/'complete.json']
        for atlas in ('a424','sch450'):
            paths += [cache_path(atlas,co),DATA/f'{co}_{atlas}.pt']
        files.update({str(p):sha(p) for p in paths})
        keep=primary_mask(m,co)
        cohorts[co]=dict(scans=int(keep.sum()),subjects=int(m.loc[keep,'subject'].nunique()),
            sufficient_80=int((keep&(m.native_frames>=SHORT)).sum()),common_no_padding_200=int((keep&(m.native_frames>=200)).sum()),
            covariates={c:int(m.loc[keep,c].notna().sum()) for c in ('age','sex','age_category','education','site','arm','native_tr','native_frames','mean_fd') if c in m})
    # Read actual selected extraction sidecars; no inference that grid resampling is registration.
    for co in ('nacc','hrv_pre'):
        source=WORK/f'data/derived/external_preliminary_v1/{co}.csv'
        m=pd.read_csv(source);selected=set(metadata(co).subject)
        for _,r in m[m.subject.isin(selected)].iterrows():
            for atlas in ('A424','Schaefer400Tian'):
                p=Path(str(r.source).replace('A424_noGSR',atlas+'_noGSR')).with_suffix('.json')
                if not p.is_absolute():p=ROOT/p
                if not p.exists():sidecars.append(dict(path=str(p),status='missing'));continue
                d=read(p);files[str(p)]=sha(p)
                sidecars.append(dict(path=str(p),cohort=co,subject=r.subject,atlas=atlas,
                    **{k:d.get(k) for k in ('shape','spatial_caveat','interpolation','script_sha256','bold','mask','global_signal_regression')},
                    source_bold_locally_available=Path(d.get('bold','/unavailable')).is_file()))
    for model in MODELS:
        version='fm_cls_posttraining_v5' if model=='brainlm13' else 'fm_posttraining_matrix_v2'
        paths=[ROOT/'runs/paper_completion_v1/canonical'/(model+'_frozen.json'),
               ROOT/'runs/fm_matched_v3/evaluation'/(atlas_for(model)+'_FC')/'adni_local.json']
        paths += [ROOT/f'runs/{version}/training/adni/{model}/supervised_{target}_s{seed}_cv/complete.json' for target in ('MEM','EF') for seed in (42,43,44)]
        files.update({str(p):sha(p) for p in paths})
    save(RUN/'spatial_audit.json',dict(status='complete',verification_status='unresolved',sidecars=sidecars,
        conclusion='Sidecar/provenance review only. Uploaded external-cohort sidecars explicitly disclaim cross-template registration. No source-volume overlay verification was possible from these records.',
        roi_order_verification='Independent correspondence of anatomical ROI labels to every tensor row and checkpoint position table remains unverified. No permutation is inferred from matching region counts.',
        required_to_resolve='Source BOLD volumes, actual atlas NIfTIs/label order and spatial transform records. Inspect registration overlays upstream. Common script/settings do not establish spatial correspondence.',
        all_cohort_claim='ADNI and original RCT spatial alignment is not independently established by this audit either. No new exclusion or ROI correction applied.'))
    save(RUN/'input_audit.json',dict(status='complete',files_sha256=files,cohorts=cohorts,
        scope='Post hoc sensitivity panel using existing eligibility. Missing covariates restrict only matched sensitivity samples; no primary-cohort exclusion.',
        clinical='HRV age categories remain categorical, without midpoint imputation. pSOPT FD is baseline only.',
        duration='80 observed frames re-centered and right-zero-padded to each native fixed input length; native TR retained. This changes information content and padding together, not an isolated TR effect.'))


def verify(paths):
    known=read(RUN/'input_audit.json')['files_sha256']
    for p in paths:
        if known.get(str(p))!=sha(p):raise ValueError('Registered input changed: '+str(p))


def shortened(x,frames=SHORT):
    if x.ndim!=3 or frames>x.shape[-1]:raise ValueError('Invalid time-window request')
    y=x[:,:,:frames].clone();y-=y.mean(-1,keepdim=True)
    return torch.nn.functional.pad(y,(0,x.shape[-1]-frames))


def extract(model,co,device='cuda'):
    from fm_readout_v4.models import Encoder
    from fm_matrix.train import amp
    verify([DATA/f'{co}_{atlas_for(model)}.pt'])
    x,_=dataset(co,model);m=pd.read_csv(RUN/'labels'/(co+'.csv'))
    ix=np.flatnonzero(primary_mask(m,co)&(m.native_frames>=SHORT))
    torch.manual_seed(0);net=Encoder(model,device);net.requires_grad_(False);net.eval()
    expected={r['model']:r['sha256'] for r in read(RUN/'pretraining_audit.json')['models']}
    if sha(net.checkpoint_path)!=expected[model]:raise ValueError('Pretrained checkpoint changed')
    batch=2 if model=='brainlm13' else 8
    values=[]
    with torch.inference_mode():
        for start in range(0,len(ix),batch):
            with amp(device):z=net(shortened(x[ix[start:start+batch]]).to(device))
            if not torch.isfinite(z).all():raise ValueError('Nonfinite short embedding')
            values.append(z.float().cpu().numpy())
            if start%(batch*20)==0:print('short_features',model,co,start,len(ix),flush=True)
    folder=RUN/'short_features'/model;folder.mkdir(parents=True,exist_ok=True)
    out=folder/(co+'.npy');temp=out.with_suffix('.tmp.npy');np.save(temp,np.concatenate(values));temp.replace(out)
    save(folder/(co+'.json'),dict(status='complete',indices=ix,scan_ids=m.iloc[ix].scan.tolist(),model=model,
        feature_sha256=sha(out),readout=net.provenance(),checkpoint_sha256=sha(net.checkpoint_path),observed_frames=SHORT,
        native_fixed_frames=net.length,encoder_training=False,batch_size=batch))


def earliest_indices(m):
    # Chronological ADNI month code, then scan ID, independent of outcomes.
    visit=m.visit.astype(str).str.upper()
    months=visit.str.extract(r'^M(\d+)$',expand=False).astype(float)
    months=months.mask(visit.isin(['BL','BASELINE','M00','M000']),0.)
    if months.isna().any():raise ValueError('Unrecognized ADNI chronological visit code')
    t=m.assign(_month=months,_ix=np.arange(len(m))).sort_values(['subject','_month','scan'])
    return t.drop_duplicates('subject')._ix.to_numpy()


def plans(co,m):
    out=[]
    basic=[c for c in ('age','sex','age_category','education') if c in m and m[c].notna().any()]
    nuisance=[c for c in ('site','arm','native_tr','native_frames','mean_fd') if c in m and m[c].notna().any()]
    for endpoint in endpoints(co):
        for setting in ('baseline','followup') if co in TRIALS else ('baseline',):
            col=endpoint if setting=='baseline' else 'follow_'+endpoint
            clinical=(['baseline_EF'] if co=='hrv_pre' else ['baseline_MEM','baseline_EF']) if setting=='followup' else []
            for mode in ('confounds','duration','no_padding')+ (('one_scan',) if co=='adni' else ()):
                b=basic+clinical if mode=='confounds' else (basic+clinical if setting=='followup' else [])
                extra=nuisance if mode=='confounds' else []
                keep=primary_mask(m,co)&np.isfinite(m[col].to_numpy(float))
                # One fixed earliest scan is selected before looking at endpoint availability.
                if mode=='one_scan':keep &= np.isin(np.arange(len(m)),earliest_indices(m))
                if b or extra:keep &= m[b+extra].notna().all(axis=1).to_numpy()
                if mode=='duration':keep &= m.native_frames.to_numpy()>=SHORT
                if mode=='no_padding':keep &= m.native_frames.to_numpy()>=200
                ix=np.flatnonzero(keep)
                out.append(dict(key=mode+'_'+endpoint+'_'+setting,mode=mode,endpoint=endpoint,setting=setting,
                    indices=ix,basic=b,nuisance=extra,budget=10 if setting=='followup' else 100,
                    y=m.iloc[ix][col].to_numpy(float),groups=m.iloc[ix].subject.astype(str).to_numpy(),
                    applicable=m.iloc[ix].subject.nunique()>=20 and m.iloc[ix][col].nunique()>1))
    return out


def covariate_pair(m,cols,tr,te):
    a=[];b=[]
    for c in cols:
        if c in ('site','arm','age_category'):
            enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
            aa=enc.fit_transform(m.iloc[tr][[c]].astype(str));bb=enc.transform(m.iloc[te][[c]].astype(str))
        else:aa=m.iloc[tr][[c]].to_numpy(float);bb=m.iloc[te][[c]].to_numpy(float)
        sc=StandardScaler().fit(aa);a.append(sc.transform(aa));b.append(sc.transform(bb))
    return (np.column_stack(a),np.column_stack(b)) if cols else (np.empty((len(tr),0)),np.empty((len(te),0)))


def prepare_fc(atlas,co,device='cuda'):
    from transfer_baseline import fc_feature
    verify([cache_path(atlas,co),RUN/'labels'/(co+'.csv'),DATA/f'{co}_{atlas}.pt'])
    m=pd.read_csv(RUN/'labels'/(co+'.csv'));fc=np.load(cache_path(atlas,co),mmap_mode='r')
    x,_=dataset(co,'brainjepa' if atlas=='sch450' else 'brainlm13')
    short=fc_feature(shortened(x).numpy(),None)
    folder=RUN/'projections'/atlas/co;folder.mkdir(parents=True,exist_ok=True);items=[];memo={}
    for plan in plans(co,m):
        key=plan['key'];items.append(key);meta=folder/(key+'.json')
        if meta.exists():continue
        if not plan['applicable']:
            save(meta,dict(status='complete',plan=plan,not_applicable='Fewer than 20 participants or constant outcome'));continue
        ix=plan['indices'];y=plan['y'];g=plan['groups'];arrays={};splits=[]
        variants={'primary':fc}
        if plan['mode']=='duration':variants['short']=short
        for seed in (0,1,2):
            for fold,(tr,te) in enumerate(grouped_splits(y,g,seed)):
                it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=g[tr]));it,iv=tr[it],tr[iv]
                prefix=f's{seed}f{fold}';splits.append(dict(seed=seed,tr=tr,te=te,it=it,iv=iv,prefix=prefix))
                for v,xx in variants.items():
                    for part,train,test in [('outer',tr,te),('inner',it,iv)]:
                        cachekey=(v,plan['budget'],tuple(ix[train]),tuple(ix[test]))
                        if cachekey not in memo:memo[cachekey]=project_block(xx[ix[train]],xx[ix[test]],plan['budget'],device)
                        aa,bb=memo[cachekey]
                        arrays[prefix+v+part+'a']=aa;arrays[prefix+v+part+'b']=bb
            print('FC projections',atlas,co,key,seed,flush=True)
        path=folder/(key+'.npz');temp=path.with_suffix('.tmp.npz');np.savez(temp,**arrays);temp.replace(path)
        save(meta,dict(status='complete',plan=plan,splits=splits,array_sha256=sha(path)))
    save(folder/'complete.json',dict(status='complete',plans=items,atlas=atlas,cohort=co))


def inference_row(y,a,b,g,family,**details):
    stats=paired(y,a,b,g,seed=digest_seed(json.dumps(details,sort_keys=True)+family))
    return dict(family=family,**details,**stats)


def evaluate(model,co):
    fm_path=feature_dir(('none',model,'frozen','none',0))/(co+'.npy')
    verify([fm_path,RUN/'labels'/(co+'.csv')])
    m=pd.read_csv(RUN/'labels'/(co+'.csv'));fm=np.load(fm_path)
    short_folder=RUN/'short_features'/model;s=read(short_folder/(co+'.json'))
    p=short_folder/(co+'.npy')
    if sha(p)!=s['feature_sha256']:raise ValueError('Short features changed')
    ff=np.load(p);lookup={int(i):k for k,i in enumerate(s['indices'])};out=RUN/'evaluation'/model/co;out.mkdir(parents=True,exist_ok=True)
    folder=RUN/'projections'/atlas_for(model)/co;memo={}
    for plan in plans(co,m):
        key=plan['key'];dest=out/(key+'.json')
        if dest.exists():continue
        proj=read(folder/(key+'.json'))
        if not plan['applicable']:save(dest,dict(status='complete',rows=[],**{k:v for k,v in proj.items() if k!='status'}));continue
        ix=plan['indices'];y=plan['y'];g=plan['groups'];mode=plan['mode']
        if proj['plan']['indices']!=ix.tolist() or proj['plan']['y']!=y.tolist():raise ValueError('Projection alignment')
        pp=folder/(key+'.npz')
        if sha(pp)!=proj['array_sha256']:raise ValueError('FC projections changed')
        fc=np.load(pp)
        recipes={'FC':('primary',None,False),'FM':(None,'primary',False)}
        if mode=='confounds':
            recipes={'basic':(None,None,False),'adjusted':(None,None,True),'basic+FC':('primary',None,False),
                'adjusted+FC':('primary',None,True),'basic+FM':(None,'primary',False),'adjusted+FM':(None,'primary',True)}
        elif mode=='duration':recipes={'FC_primary':('primary',None,False),'FC_short':('short',None,False),'FM_primary':(None,'primary',False),'FM_short':(None,'short',False)}
        prepared={r:{seed:[] for seed in (0,1,2)} for r in recipes}
        for split in proj['splits']:
            tr,te,it,iv=[np.asarray(split[c],int) for c in ('tr','te','it','iv')];seed=split['seed'];prefix=split['prefix']
            transformed={}
            for v,x in [('primary',fm[ix])]+([('short',ff[[lookup[int(i)] for i in ix]])] if mode=='duration' else []):
                transformed[v]=[]
                for a,b in [(tr,te),(it,iv)]:
                    cachekey=(v,plan['budget'],tuple(ix[a]),tuple(ix[b]))
                    if cachekey not in memo:memo[cachekey]=project_block(x[a],x[b],plan['budget'])
                    transformed[v].append(memo[cachekey])
            cbase=[covariate_pair(m,plan['basic'],ix[a],ix[b]) for a,b in [(tr,te),(it,iv)]]
            cextra=[covariate_pair(m,plan['nuisance'],ix[a],ix[b]) for a,b in [(tr,te),(it,iv)]]
            for name,(vf,vm,adjust) in recipes.items():
                pairs=[]
                for num,part in enumerate(('outer','inner')):
                    pair=[]
                    for j in (0,1):
                        blocks=[cbase[num][j]]
                        if adjust:blocks.append(cextra[num][j])
                        if vf:blocks.append(fc[prefix+vf+part+('a' if j==0 else 'b')])
                        if vm:blocks.append(transformed[vm][num][j])
                        pair.append(np.column_stack(blocks))
                    pairs.append(tuple(pair))
                prepared[name][seed].append((tr,te,pairs[0],it,iv,pairs[1]))
        pred={name:np.mean([predict_prepared(by[s],y,binary=False) for s in (0,1,2)],axis=0) for name,by in prepared.items()}
        if mode=='confounds':pairs=[('basic+FC','basic+FM'),('adjusted+FC','adjusted+FM'),('basic','basic+FM'),('adjusted','adjusted+FM'),('adjusted','adjusted+FC')]
        elif mode=='duration':pairs=[('FC_primary','FC_short'),('FM_primary','FM_short'),('FC_primary','FM_primary'),('FC_short','FM_short')]
        else:pairs=[('FC','FM')]
        rows=[]
        for a,b in pairs:
            family='_'.join([mode,plan['endpoint'],plan['setting'],a+'_vs_'+b])
            rows.append(inference_row(y,pred[a],pred[b],g,family,model=model,cohort=co,endpoint=plan['endpoint'],setting=plan['setting'],reference=a,candidate=b))
        save(dest,dict(status='complete',rows=rows,plan=plan,scan_ids=m.iloc[ix].scan.tolist(),predictions=pred,
            split_audit=proj['splits'],protocol='Nested grouped five-fold CV, 20% inner validation, seeds 0/1/2. All scaling, categorical encoding and PCA fitted inside training partitions. FC/FM rank <=100 baseline or <=10 follow-up. Same participants and folds for every contrast.'))
        print('evaluated',model,co,key,len(ix),flush=True)
    save(out/'complete.json',dict(status='complete',model=model,cohort=co))


def weighted_pair(y,a,b,g,seed=0,n_boot=5000):
    y,a,b=np.asarray(y),np.asarray(a),np.asarray(b);subjects,inv,counts=np.unique(g,return_inverse=True,return_counts=True)
    base=1/counts[inv];sa=weighted_scores(y,a,base[None])[0];sb=weighted_scores(y,b,base[None])[0]
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,n_boot,128):
        w=rng.multinomial(len(subjects),np.full(len(subjects),1/len(subjects)),size=min(128,n_boot-start))[:,inv]*base
        draws.extend(weighted_scores(y,b,w)-weighted_scores(y,a,w))
    draws=np.asarray(draws);valid=draws[np.isfinite(draws)]
    if len(valid)<.95*n_boot:raise ValueError('Invalid weighted bootstrap replicates')
    return dict(reference_score=sa,candidate_score=sb,delta=sb-sa,ci95=np.quantile(valid,[.025,.975]),n=len(y),n_subjects=len(subjects),
        estimand='Weighted Spearman: each participant has total weight one across visits. Both outcomes and predictions are ranked using these weights.',
        uncertainty='Participant-cluster bootstrap of fixed OOF predictions, 5000 draws. No refitting, permutation p-value, or equivalence claim.')


def weighting(model):
    rows=[]
    # Source-CV predictions are from separately fitted fold-specific encoders.
    for target in ('MEM','EF'):
        version='fm_cls_posttraining_v5' if model=='brainlm13' else 'fm_posttraining_matrix_v2';frames=[]
        for seed in (42,43,44):
            p=ROOT/f'runs/{version}/training/adni/{model}/supervised_{target}_s{seed}_cv/complete.json'
            verify([p])
            folds=read(p)['folds']
            for f in folds:
                if set(f['test_subjects'])&(set(f['train_subjects'])|set(f['validation_subjects'])):raise ValueError('CV overlap')
            frame=pd.DataFrame({k:np.concatenate([f[k] for f in folds]) for k in ('test_indices','test_subjects','y','frozen_prediction','adapted_prediction')}).sort_values('test_indices')
            frames.append(frame)
        for f in frames[1:]:
            if not f[['test_indices','test_subjects','y']].reset_index(drop=True).equals(frames[0][['test_indices','test_subjects','y']].reset_index(drop=True)):raise ValueError('Weighted alignment')
        f=frames[0];a=np.mean([z.frozen_prediction for z in frames],axis=0);b=np.mean([z.adapted_prediction for z in frames],axis=0)
        rows.append(dict(model=model,endpoint=target,reference='frozen_source_CV',candidate='supervised_source_CV',
            session_weighted_delta=score(f.y,b,False)-score(f.y,a,False),**weighted_pair(f.y.to_numpy(),a,b,f.test_subjects.to_numpy(),digest_seed(model+target))))
    # Within-cohort frozen FM vs atlas-matched FC, using saved main-analysis OOF predictions.
    p=ROOT/'runs/paper_completion_v1/canonical'/(model+'_frozen.json');verify([p]);main=read(p)
    candidates=main.get('rows',main.get('local',[]))
    # Actual canonical schema is checked rather than assuming row order.
    if not isinstance(candidates,list):raise ValueError('Unrecognized canonical local rows')
    for r in candidates:
        if r['cohort']!='adni' or r['method']!='FM' or not r['task'].endswith('_baseline'):continue
        p=ROOT/'runs/fm_matched_v3/evaluation'/(atlas_for(model)+'_FC')/'adni_local.json'
        verify([p])
        fs=[z for z in read(p)['rows'] if z['task']==r['task'] and z['method']=='FC']
        if len(fs)!=1:raise ValueError('Missing FC weighted reference')
        ref=fs[0]
        if ref['subjects']!=r['subjects'] or not np.allclose(ref['y'],r['y']):raise ValueError('FC weighted alignment')
        rows.append(dict(model=model,endpoint=r['task'],reference='FC',candidate='FM',
            session_weighted_delta=score(r['y'],r['prediction'],False)-score(ref['y'],ref['prediction'],False),
            **weighted_pair(r['y'],ref['prediction'],r['prediction'],r['subjects'],digest_seed(model+r['task']))))
    if len(rows)!=4:raise ValueError('Expected four weighted contrasts')
    save(RUN/'weighting'/(model+'.json'),dict(status='complete',rows=rows))


def report():
    rows=[];not_applicable=[]
    for p in sorted((RUN/'evaluation').glob('*/*/*.json')):
        d=read(p)
        for r in d.get('rows',[]):rows.append(dict(r,input_file=str(p)))
        if 'not_applicable' in d:not_applicable.append(dict(path=str(p),reason=d['not_applicable']))
    families={}
    for r in rows:families.setdefault(r['family'],[]).append(r)
    for name,group in families.items():
        for r,p in zip(group,holm([r['permutation_p'] for r in group])):r.update(holm_p=p,family_size=len(group))
        if len(group)/20001>.05:raise ValueError('Insufficient permutation resolution')
    weighted=[r for p in (RUN/'weighting').glob('*.json') for r in read(p)['rows']]
    pd.json_normalize(rows).to_csv(RUN/'comparisons.csv',index=False)
    pd.json_normalize(weighted).to_csv(RUN/'participant_weighted.csv',index=False)
    save(RUN/'report.json',dict(status='complete',comparisons=rows,participant_weighted=weighted,not_applicable=not_applicable,
        n_comparisons=len(rows),families={k:len(v) for k,v in families.items()},
        interpretation='Post hoc conditional robustness evidence; no training-algorithm superiority or equivalence claim. Restricting to unpadded scans changes the population; do not interpret its score difference from the full cohort as a causal padding effect.',
        provenance_audits=dict(spatial=read(RUN/'spatial_audit.json'),pretraining=read(RUN/'pretraining_audit.json')),
        automatic_shutdown=False))
