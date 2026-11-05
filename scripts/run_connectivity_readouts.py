#!/usr/bin/env python3
"""Additive, frozen final-layer spatial embedding-similarity benchmark.

No parent results, main readouts, encoder weights, or manuscript are modified.
Use `run --device cpu` for cached CPU work and explicitly allowed CPU extraction;
GPU extraction jobs stay pending when CUDA is unavailable.
"""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import hashlib
import json
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from fm_matrix.data import save, sha

RUN = ROOT/'runs/connectivity_readouts_v2'
OUT = ROOT/'results/generated/connectivity_readouts_v2'
MODELS = ('brainlm13','brainlm111','brainlm650','brainjepa','semantoks',
          'fmrilm','fmrilm_mae','fmrilm_vq_domain','brainharmonix_f')
COHORTS = ('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post','nacc_repeated')
PAIRS = ('adni','nacc_repeated','act','cogte','breathe','psopt','hrv')
CACHED = ('brainjepa','semantoks')
DEFINITIONS = {
    'brainlm13': '424 ROI summaries, temporal mean of final normalized content tokens; 89,676 edges. Also export layer3 and raw layer4 for a layer-only contrast.',
    'brainlm111': '27 spatial patch-row summaries; average temporal patch columns weighted by overlap with the 200-frame input. 351 edges. These are not individual ROI embeddings.',
    'brainlm650': '31 spatial patch-row summaries; average temporal patch columns weighted by overlap with the 200-frame input. 465 edges. These are not individual ROI embeddings.',
    'brainjepa': '450 ROI summaries from cached final tokens averaged over ten temporal tokens; 101,025 edges.',
    'semantoks': '8 observed network summaries from cached network_flat, excluding unavailable cerebellar network; 28 edges. These are not individual ROI embeddings.',
    'fmrilm': '450 final stage2 brain-token summaries, each corresponding to an ROI; 101,025 edges.',
    'fmrilm_mae': '450 final stage2 brain-token summaries, each corresponding to an ROI; 101,025 edges.',
    'fmrilm_vq_domain': '450 final stage2 brain-token summaries, each corresponding to an ROI; 101,025 edges.',
    'brainharmonix_f': '400 ROI summaries, pooling valid final normalized temporal tokens only; 79,800 edges. Extract regional_flat and all other readouts in the same forward pass.',
}

def readout_names(model):
    names=['native','spatial_flat','spatial_norm','connectivity']
    if model in ('brainlm13','brainlm111','brainlm650','semantoks'):names+=['content_mean']
    if model=='semantoks':names+=['cls_mean_concat']
    if model=='brainharmonix_f':names+=['token_std']
    return names

def regional_readouts(regions):
    regions=np.asarray(regions,dtype=np.float32)
    return dict(spatial_flat=regions.reshape(len(regions),-1),
                spatial_norm=np.linalg.norm(regions,axis=-1),connectivity=connectivity(regions))

def read(path):
    return json.loads(Path(path).read_text())

def seed(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)

def parent_folder(model):
    from unified_supplements import feature_folder
    if model=='brainharmonix_f':
        return ROOT/'runs/brainharmonix_f_v1/features/brainharmonix_f_frozen'
    return feature_folder(('none',model,'frozen','none',0))

def connectivity(regions):
    """Pearson similarity across feature channels, not across BOLD time points."""
    z=np.asarray(regions,dtype=np.float32)
    if z.ndim!=3 or not np.isfinite(z).all():
        raise ValueError('Expected finite batch x spatial units x feature channels')
    z=z-z.mean(-1,keepdims=True)
    z=z/np.maximum(np.linalg.norm(z,axis=-1,keepdims=True),1e-8)
    a=z@z.transpose(0,2,1)
    i,j=np.triu_indices(z.shape[1],1)
    return a[:,i,j]

def spatial_patch_mean(tokens,image_size,patch_size,rois=424,frames=200):
    """Observed-input temporal overlap weighting; never invent per-ROI tokens."""
    size=image_size[0] if isinstance(image_size,(tuple,list)) else image_size
    patch=patch_size[0] if isinstance(patch_size,(tuple,list)) else patch_size
    grid=size//patch
    z=np.asarray(tokens).reshape(len(tokens),grid,grid,-1)
    left=(size-frames)//2;top=(size-rois)//2
    starts=np.arange(grid)*patch
    w=np.maximum(0,np.minimum(starts+patch,left+frames)-np.maximum(starts,left))
    keep=np.minimum(starts+patch,top+rois)>np.maximum(starts,top)
    return np.einsum('brtd,t->brd',z[:,keep],w/w.sum()).astype(np.float32)

def init():
    RUN.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for model in MODELS:
        for co in COHORTS:
            tasks.append(dict(id=f'extract/{model}/{co}',action='extract',model=model,cohort=co,
                deps=[],cached=model in CACHED,output=f'features/{model}/{co}/complete.json'))
        for co in ('adni','nacc'):
            for ep in ('MEM','EF'):
                tasks.append(dict(id=f'cognition/{model}/{co}/{ep}',action='cognition',model=model,cohort=co,endpoint=ep,
                    deps=[f'extract/{model}/{co}'],output=f'cognition/{model}/{co}_{ep}.json'))
        for co in PAIRS:
            datasets=('hrv_pre','hrv_post') if co=='hrv' else (co,)
            tasks.append(dict(id=f'identity/{model}/{co}',action='identity',model=model,cohort=co,
                deps=[f'extract/{model}/{c}' for c in datasets],output=f'identity/{model}/{co}.json'))
    d=dict(version=2,tasks=tasks,definitions=DEFINITIONS,readouts={m:readout_names(m) for m in MODELS},code_sha256=sha(Path(__file__)),
        encoder_training=False,automatic_shutdown=False,manuscript_editing=False,
        scope='Retrospective frozen readout sensitivity. All nine configurations; ADNI/NACC MEM/EF; common-sample identification in seven cohorts. Not a new all-task or post-training queue.',
        inputs='Unchanged model-specific input lengths, atlas mapping, normalization and missing-input masks.',
        cognition='Reuse SF100 task scan IDs, targets and all 15 grouped outer/inner partitions. Training-only standardization/exact PCA<=100, unchanged ridge grid. Average three split seeds before scoring.',
        identification='Reuse common paired scans. Correlation similarity, bidirectional fractional tie credit. Native dimensions and fixed Gaussian projection to128 (seeds0/1/2). Conditional on fixed gallery.',
        inference='5000 participant-cluster bootstraps,20000 two-sided conditional swaps. Holm across ALL supported final-layer readouts, nine configurations and cohorts separately by endpoint/metric, identity mode and comparator. Layer3/raw-layer4 contrast separate. No new absolute null tests.',
        references='Same-pass primary readout for new inference; immutable cached primary for cached summaries. Schaefer100 FC. BrainLM13 same-pass raw layer3 and raw layer4 isolate layer, separately from final LayerNorm.',
        precision='New inference uses bfloat16 autocast on CUDA or CPU, with float32 similarity construction. Same-pass primary controls and comparisons against archived primary embeddings audit numerical drift.',
        layer_policy='Every primary and alternative readout in the main comparison uses the final encoder output INCLUDING the official final normalization. fMRI-LM image_encoder_mean is a different stage and excluded. No layer sweep or outcome-based selection. 13M raw L3/raw L4/normalized L4 contrasts are separate exploratory families.',
        sources=dict(parent_comparisons_sha256=sha(ROOT/'results/generated/brainharmonix_f_v1/joint/Comparisons.csv')))
    p=RUN/'manifest.json'
    if p.exists():
        if read(p)!=d:raise ValueError('Registered experiment changed; do not overwrite an existing run')
    else:save(p,d)
    return tasks

def extract(model,co,device='cpu',smoke=False):
    import torch
    torch.set_num_threads(2)
    from fm_readout_v4.data import metadata
    dest=RUN/'features'/model/co
    if (dest/'complete.json').exists() and not smoke:return
    m=metadata(co);parent=parent_folder(model);reference=np.load(parent/(co+'.npy'),mmap_mode='r')
    if len(reference)!=len(m):raise ValueError('Primary row count mismatch')
    sources={str(parent/(co+'.npy')):sha(parent/(co+'.npy')),str(parent/'complete.json'):sha(parent/'complete.json')}
    arrays={};start=time.time()
    if model in CACHED:
        if model=='brainjepa':
            p=ROOT/'runs/readout_sensitivity_v1/roi_features'/co/'roi.npy'
            stamp=read(p.parent/'complete.json')
            if stamp['scan_ids']!=m.scan.tolist() or stamp['feature_sha256']['roi.npy']!=sha(p):
                raise ValueError('JEPA regional feature provenance mismatch')
            regions=np.load(p,mmap_mode='r')
        else:
            p=parent/(co+'_alternatives.npz')
            with np.load(p) as a:
                regions=a['network_flat'].reshape(len(m),8,768)
                arrays['content_mean']=a['content_mean']
                arrays['cls_mean_concat']=a['cls_avg']
        sources[str(p)]=sha(p)
        arrays.update(spatial_flat=regions.reshape(len(regions),-1),spatial_norm=np.linalg.norm(regions,axis=-1),
            connectivity=np.concatenate([connectivity(regions[i:i+8]) for i in range(0,len(regions),8)]))
        arrays['native']=np.asarray(reference)
    else:
        if model in ('brainlm13','brainlm111','brainlm650'):
            from fm_readout_v4.models import Encoder
            from fm_readout_v4.data import dataset,DATA
            x,mm=dataset(co,model);input_path=DATA/(co+'_a424.pt')
            net=Encoder(model,device)
        elif model=='brainharmonix_f':
            from brainharmonix.models import Encoder
            from brainharmonix.data import dataset,DATA
            x,mm=dataset(co,model);input_path=DATA/(co+'_'+model+'.pt');net=Encoder(model,device)
        else:
            from new_fm.data import dataset,DATA
            x,mm=dataset(co,'fmrilm');input_path=DATA/(co+'_fmrilm.pt')
            if model=='fmrilm':
                from new_fm.models import Encoder
                net=Encoder('fmrilm',device)
            else:
                from fmrilm_variants.runtime import configure
                _,Encoder=configure('mae' if model=='fmrilm_mae' else 'vq-domain')
                net=Encoder('fmrilm',device)
        if mm.scan.tolist()!=m.scan.tolist():raise ValueError('Input row order mismatch')
        sources[str(input_path)]=sha(input_path)
        sources[str(net.checkpoint_path)]=sha(net.checkpoint_path)
        net.eval().requires_grad_(False)
        batch=4 if model.startswith('fmrilm') else 1
        n=min(max(2,batch),len(x)) if smoke else len(x)
        # Resumable batches; incomplete exports never become eligible evaluation inputs.
        dest.mkdir(parents=True,exist_ok=True)
        chunks=dest/'smoke'/sha(Path(__file__)) if smoke else dest/'chunks'
        chunks.mkdir(parents=True,exist_ok=True)
        signature=dict(sources=sources,device=device,batch_size=batch,code_sha256=sha(Path(__file__)),readouts=readout_names(model))
        request=chunks/'request.json'
        if request.exists() and read(request)!=signature:raise ValueError('Partial extraction settings changed')
        if not request.exists():save(request,signature)
        for i in range(0,n,batch):
            end=min(i+batch,n)
            if all((chunks/f'{j:05d}.npz').exists() for j in range(i,end)):continue
            with torch.inference_mode(),torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):
                xx=x[i:end].to(device)
                if model=='brainlm13':
                    o=net.net.vit(signal_vectors=xx,xyz_vectors=net.xyz[None],noise=None,output_hidden_states=True,return_dict=True)
                    native=o.last_hidden_state[:,0].float().cpu().numpy()
                    tokens=o.last_hidden_state[:,1:].float().reshape(1,424,10,-1).mean(2).cpu().numpy()
                    l3=o.hidden_states[3][:,1:].float().reshape(1,424,10,-1).mean(2).cpu().numpy()
                    l4=o.hidden_states[4][:,1:].float().reshape(1,424,10,-1).mean(2).cpu().numpy()
                elif model.startswith('brainlm'):
                    z=net.tokens(xx);native=z[:,0].float().cpu().numpy()
                    tokens=spatial_patch_mean(z[:,1:].float().cpu().numpy(),net.cfg.image_size,net.cfg.patch_size)
                elif model=='brainharmonix_f':
                    z,mask=net.token_outputs(xx);z=z.float()
                    primary=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
                    native=primary.cpu().numpy()
                    tokens=(z.reshape(1,400,18,768).sum(2)/mask.reshape(1,400,18).sum(2)[:,:,None]).cpu().numpy()
                    std=torch.sqrt(((z-primary[:,None]).square()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy()
                else:
                    z=net.tokens(xx);native=z.mean(1).float().cpu().numpy();tokens=z.float().cpu().numpy()
            vals=dict(native=native,**regional_readouts(tokens))
            if model.startswith('brainlm'):
                if model=='brainlm13':vals['content_mean']=tokens.mean(1)
                else:
                    from fm_matched_v3.models import patch_coverage
                    w=patch_coverage(424,200,net.cfg.image_size,net.cfg.patch_size)
                    vals['content_mean']=np.einsum('bnd,n->bd',z[:,1:].float().cpu().numpy(),w/w.sum()).astype(np.float32)
            if model=='brainharmonix_f':vals['token_std']=std
            if model=='brainlm13':vals.update(layer3=connectivity(l3),layer4_raw=connectivity(l4))
            if not all(np.isfinite(a).all() for a in vals.values()):raise ValueError('Nonfinite extraction')
            for j in range(i,end):
                path=chunks/f'{j:05d}.npz';tmp=path.with_suffix('.tmp.npz')
                np.savez(tmp,**{k:v[j-i:j-i+1] for k,v in vals.items()});tmp.replace(path)
            if i%25==0:print('extract',model,co,i,n,'elapsed',round(time.time()-start,1),flush=True)
        # Stream large regional-flat exports rather than retaining all scan chunks.
        with np.load(chunks/'00000.npz') as first:
            arrays={k:np.lib.format.open_memmap(dest/(k+('.smoke' if smoke else '')+'.building.npy'),mode='w+',
                dtype=np.float32,shape=(n,*first[k].shape[1:])) for k in first.files}
        for i in range(n):
            with np.load(chunks/f'{i:05d}.npz') as a:
                for k in a.files:arrays[k][i:i+1]=a[k]
        for a in arrays.values():a.flush()
    ref=np.asarray(reference[:len(arrays['native'])],float);new=np.asarray(arrays['native'],float)
    cos=np.sum(ref*new,axis=1)/np.maximum(np.linalg.norm(ref,axis=1)*np.linalg.norm(new,axis=1),1e-12)
    drift=dict(min_cosine=float(cos.min()),max_absolute_error=float(np.max(abs(ref-new))),
        relative_rmse=float(np.sqrt(np.mean((ref-new)**2))/max(np.sqrt(np.mean(ref**2)),1e-12)))
    if drift['min_cosine']<.9999 or drift['relative_rmse']>.005:
        raise ValueError('Primary readout drift exceeds audit tolerance: '+str(drift))
    if set(arrays)!=(set(readout_names(model))|({'layer3','layer4_raw'} if model=='brainlm13' else set())):raise ValueError('Missing or extraneous readouts')
    stamp=dict(status='complete',model=model,cohort=co,scan_ids=m.scan.tolist(),sources_sha256=sources,
        layer='final encoder output after official final normalization; raw layer3/layer4 separately labeled',
        definition=DEFINITIONS[model],device=device,elapsed_seconds=time.time()-start,primary_reproduction=drift)
    if smoke:
        stamp.update(status='smoke_only',tested_scans=len(new),shapes={k:list(v.shape) for k,v in arrays.items()})
        save(RUN/'smoke'/(model+'_'+co+'.json'),stamp)
        print(json.dumps({k:stamp[k] for k in ('model','cohort','status','elapsed_seconds','primary_reproduction','tested_scans','shapes')}),flush=True);return
    dest.mkdir(parents=True,exist_ok=True)
    for k,a in arrays.items():
        p=dest/(k+'.npy')
        if isinstance(a,np.memmap) and Path(a.filename)==dest/(k+'.building.npy'):
            a.flush();Path(a.filename).replace(p)
        else:
            tmp=p.with_suffix('.tmp.npy');np.save(tmp,a);tmp.replace(p)
    stamp['features_sha256']={k:sha(dest/(k+'.npy')) for k in arrays}
    save(dest/'complete.json',stamp)

def features(model,co):
    from fm_readout_v4.data import metadata
    folder=RUN/'features'/model/co;d=read(folder/'complete.json')
    if d['scan_ids']!=metadata(co).scan.tolist():raise ValueError('Feature row mismatch')
    out={}
    for k,h in d['features_sha256'].items():
        p=folder/(k+'.npy')
        if sha(p)!=h:raise ValueError('Feature changed')
        out[k]=np.load(p,mmap_mode='r')
    return out

def contrasts(model):
    pairs=[]
    for name in readout_names(model):
        if name=='native':continue
        pairs.extend([('native',name,'final_readouts'),('sf100_fc',name,'final_readouts_fc')])
    if model=='brainlm13':pairs += [('layer3','layer4_raw','layer_only'),('layer4_raw','connectivity','final_norm')]
    return pairs

def project_indices(x,tr,te,k=100,device='cpu'):
    """Train-only exact PCA using a streamed double-precision Gram matrix.

    Same StandardScaler and PCA as the parent, without materializing a multi-GB
    standardized feature matrix. GPU accelerates only label-free matrix algebra.
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.linalg import eigh
    from paper_completion import project_block
    if x.shape[1]<5000:return project_block(x[tr],x[te],k)
    import torch
    gpu=device.startswith('cuda');n=len(tr);k=min(k,n-1,x.shape[1])
    gram=torch.zeros((n,n),dtype=torch.float64,device=device) if gpu else np.zeros((n,n))
    cross=torch.zeros((len(te),n),dtype=torch.float64,device=device) if gpu else np.zeros((len(te),n))
    for lo in range(0,x.shape[1],8192):
        hi=min(lo+8192,x.shape[1]);block=np.asarray(x[:,lo:hi],dtype=np.float64)
        sc=StandardScaler().fit(block[tr]);a=sc.transform(block[tr]);b=sc.transform(block[te])
        # PCA centers the standardized training matrix once more.
        center=a.mean(0);a-=center;b-=center
        if gpu:
            aa=torch.as_tensor(a,device=device);bb=torch.as_tensor(b,device=device)
            gram.addmm_(aa,aa.T);cross.addmm_(bb,aa.T)
        else:gram+=a@a.T;cross+=b@a.T
    if gpu:gram=gram.cpu().numpy();cross=cross.cpu().numpy()
    e,u=eigh(gram,subset_by_index=(n-k,n-1),check_finite=False);e=e[::-1].clip(min=0);u=u[:,::-1]
    tol=np.finfo(float).eps*max(n,x.shape[1])*max(float(e.max()),1)
    inv=np.divide(1,np.sqrt(np.maximum(e,tol)),out=np.zeros_like(e),where=e>tol)
    return (gram@u)*inv,(cross@u)*inv

def cognition(model,co,ep,device='cpu'):
    import sf100_benchmark as sf
    from benchmark_evaluation import predict_prepared,score
    from paper_paired_stats import paired
    template=sf.RUN/'evaluation'/co/(ep+'.json');old=read(template);m=sf.labels(co)
    ix=m.set_index('scan').index.get_indexer(old['scan_ids'])
    if (ix<0).any() or m.iloc[ix].subject.astype(str).tolist()!=old['subjects']:raise ValueError('Sample mismatch')
    y=np.asarray(old['y']);g=np.asarray(old['subjects']);xs=features(model,co);pp={};byseed={}
    for name,x in xs.items():
        prepared={s:[] for s in (0,1,2)}
        for f in old['split_audit']:
            tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
            if set(g[tr])&set(g[te]) or set(g[it])&set(g[iv]):raise ValueError('Participant leakage')
            prepared[f['seed']].append((tr,te,project_indices(x,ix[tr],ix[te],100,device),it,iv,project_indices(x,ix[it],ix[iv],100,device)))
        byseed[name]=[predict_prepared(prepared[s],y,binary=False) for s in (0,1,2)]
        pp[name]=np.mean(byseed[name],axis=0)
        print('cognition',model,co,ep,name,score(y,pp[name],False),flush=True)
    pp['sf100_fc']=np.asarray(old['predictions']['sf100_fc'])
    rows=[]
    for ref,cand,fam in contrasts(model):
        stats=paired(y,pp[ref],pp[cand],g,seed=seed(model+co+ep+fam))
        rows.append(dict(model=model,cohort=co,task=ep,reference=ref,candidate=cand,
            family=f'connectivity::{fam}::{ep}',**stats))
    save(RUN/'cognition'/model/(co+'_'+ep+'.json'),dict(status='complete',rows=rows,scan_ids=old['scan_ids'],subjects=g,
        y=y,predictions=pp,predictions_by_seed=byseed,scores={k:score(y,p,False) for k,p in pp.items()},
        split_audit=old['split_audit'],template_sha256=sha(template),feature_dimensions={k:x.shape[1] for k,x in xs.items()}))

def identity(model,co):
    import sample_alignment as sa
    import sf100_benchmark as sf
    import readout_sensitivity as rs
    from unified_supplements import projected_similarity
    subjects,(ca,ia),(cb,ib)=sa.paired_rows(co);a=features(model,ca);b=a if ca==cb else features(model,cb)
    aa={k:v[ia] for k,v in a.items()};bb={k:v[ib] for k,v in b.items()}
    aa['sf100_fc']=sf.features(ca,include_fm=False)['sf100_fc'][ia]
    bb['sf100_fc']=sf.features(cb,include_fm=False)['sf100_fc'][ib]
    rows=[];scores={}
    for mode in ('native','projected128'):
        fn=rs.similarity if mode=='native' else projected_similarity
        scores[mode]={k:rs.identification_scores(fn(v,bb[k])) for k,v in aa.items()}
        for ref,cand,fam in contrasts(model):
            for metric in ('top1','I_diff'):
                stats=rs.paired_identification(scores[mode][ref]['per_subject_'+metric],scores[mode][cand]['per_subject_'+metric],seed(model+co+mode+metric+fam))
                rows.append(dict(model=model,cohort=co,task='identification',mode=mode,metric=metric,reference=ref,candidate=cand,
                    family=f'connectivity::{fam}::{mode}::{metric}',**stats))
    save(RUN/'identity'/model/(co+'.json'),dict(status='complete',rows=rows,scores=scores,subjects=subjects,
        reference='Common paired scans; no participant fitted projection; fixed-gallery inference',absolute_nulls=False))

def report(tasks):
    from paper_paired_stats import holm
    rows=[];coverage=[];scores=[]
    for t in tasks:
        p=RUN/t['output'];done=p.exists()
        coverage.append(dict(id=t['id'],status='complete' if done else 'pending'))
        if done and t['action']!='extract':
            d=read(p);rows.extend(d['rows'])
            if t['action']=='cognition':
                scores.extend(dict(model=t['model'],cohort=t['cohort'],task=t['endpoint'],readout=k,metric='rho',score=v,
                    n=len(d['subjects']),participants=len(set(d['subjects']))) for k,v in d['scores'].items())
            else:
                scores.extend(dict(model=t['model'],cohort=t['cohort'],task='identification',readout=k,mode=mode,metric=metric,score=v[metric],
                    n=len(d['subjects']),participants=len(d['subjects'])) for mode,ss in d['scores'].items() for k,v in ss.items() for metric in ('top1','I_diff'))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(coverage).to_csv(OUT/'Coverage.csv',index=False)
    d=pd.DataFrame(rows)
    complete=all(r['status']=='complete' for r in coverage)
    if len(d):
        # Do not display partially corrected P values as a final nine-model result.
        d['holm_p']=np.nan
        if complete:
            for _,ix in d.groupby('family').groups.items():d.loc[ix,'holm_p']=holm(d.loc[ix,'permutation_p'].fillna(1).tolist())
        d['inference_status']='final_nine_configuration_family' if complete else 'pending_complete_family'
        d.to_csv(OUT/'Comparisons.csv',index=False)
        pd.DataFrame(scores).to_csv(OUT/'Scores.csv',index=False)
        with pd.ExcelWriter(OUT/'Connectivity_readouts.xlsx',engine='xlsxwriter') as w:
            d.to_excel(w,sheet_name='Comparisons',index=False)
            pd.DataFrame(scores).to_excel(w,sheet_name='Scores',index=False)
            pd.DataFrame(coverage).to_excel(w,sheet_name='Coverage',index=False)
            pd.DataFrame([dict(model=k,definition=v) for k,v in DEFINITIONS.items()]).to_excel(w,sheet_name='Definitions',index=False)
    status=dict(total=len(tasks),complete=sum(x['status']=='complete' for x in coverage),pending=sum(x['status']!='complete' for x in coverage),all_complete=complete,
        automatic_shutdown=False,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    save(RUN/'summary.json',status)
    if complete:save(RUN/'complete.json',status)
    return status

def run(device,cpu_extract,workers,gpu_workers=2):
    import fcntl
    import torch
    tasks=init();lock=open(RUN/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};failed=set();logs=ROOT/'logs/connectivity_readouts_v2';logs.mkdir(parents=True,exist_ok=True)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested GPU not visible; no silent CPU fallback')
    while True:
        done={t['id'] for t in tasks if (RUN/t['output']).exists()}
        for key,(proc,t,fh) in list(active.items()):
            if proc.poll() is None:continue
            fh.close();del active[key]
            if proc.returncode or not (RUN/t['output']).exists():failed.add(key)
            print('finished',key,'exit',proc.returncode,flush=True)
        # Newly ready analyses start before further extraction to avoid a stage barrier.
        pending=sorted(tasks,key=lambda t:(t['action']=='extract',t['model']!='brainlm13'))
        for t in pending:
            if t['id'] in done or t['id'] in active or t['id'] in failed or not set(t['deps'])<=done:continue
            needs_encoder=t['action']=='extract' and not t['cached']
            needs_gpu=device.startswith('cuda') and (needs_encoder or t['action']=='cognition')
            active_gpu=sum(v[1].get('on_gpu',False) for v in active.values())
            if needs_gpu and active_gpu>=gpu_workers:continue
            if not needs_gpu and len(active)-active_gpu>=workers:continue
            # Reserve one GPU lane for inference while both kinds of work are ready.
            if needs_gpu and t['action']=='cognition' and any(v[1]['action']=='cognition' for v in active.values()):
                if any(j['action']=='extract' and not j['cached'] and j['id'] not in done|set(active)|failed for j in tasks):continue
            if needs_encoder and device=='cpu' and t['model'] not in cpu_extract:continue
            if needs_encoder and device.startswith('cuda') and not torch.cuda.is_available():continue
            cmd=[sys.executable,str(Path(__file__).resolve()),t['action'],'--model',t['model'],'--cohort',t['cohort'],'--device',device if needs_gpu or needs_encoder else 'cpu']
            if 'endpoint' in t:cmd+=['--endpoint',t['endpoint']]
            fh=open(logs/(t['id'].replace('/','_')+'.log'),'a')
            active[t['id']]=(subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT),dict(t,on_gpu=needs_gpu),fh)
            print('started',t['id'],flush=True)
        save(RUN/'status.json',dict(completed=len(done),total=len(tasks),running=list(active),failed=sorted(failed),device=device,
            cuda_available=torch.cuda.is_available(),pending=len(tasks)-len(done)-len(active),automatic_shutdown=False,
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        if not active:break
        time.sleep(5)
    print(json.dumps(report(tasks)),flush=True)

def validate(device):
    """Check every architecture in a separate process before scheduling the cohort run."""
    from sklearn.linear_model import Ridge
    rng=np.random.default_rng(42);x=rng.normal(size=(37,5200));tr=np.arange(27);te=np.arange(27,37);y=rng.normal(size=len(tr))
    a,b=project_indices(x,tr,te,8,'cpu');c,d=project_indices(x,tr,te,8,device)
    np.testing.assert_allclose(Ridge(alpha=10).fit(a,y).predict(b),Ridge(alpha=10).fit(c,y).predict(d),rtol=1e-6,atol=1e-8)
    for model in MODELS:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'smoke','--model',model,'--cohort','psopt','--device',device],check=True)
    save(RUN/'validation.json',dict(status='passed',models=list(MODELS),device=device,
        streamed_pca_gpu_cpu_prediction_equivalence=True,code_sha256=sha(Path(__file__))))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','extract','cognition','identity','run','report','smoke','validate'])
    p.add_argument('--model',choices=MODELS);p.add_argument('--cohort',choices=(*COHORTS,'hrv'))
    p.add_argument('--endpoint',choices=['MEM','EF']);p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-extract',nargs='*',default=[]);p.add_argument('--workers',type=int,default=2);p.add_argument('--gpu-workers',type=int,default=2)
    a=p.parse_args()
    if a.action=='validate':validate(a.device)
    elif a.action=='init':print(len(init()))
    elif a.action in ('extract','smoke'):extract(a.model,a.cohort,a.device,a.action=='smoke')
    elif a.action=='cognition':cognition(a.model,a.cohort,a.endpoint,a.device)
    elif a.action=='identity':identity(a.model,a.cohort)
    elif a.action=='report':print(report(read(RUN/'manifest.json')['tasks']))
    else:run(a.device,a.cpu_extract,a.workers,a.gpu_workers)

if __name__=='__main__':main()
