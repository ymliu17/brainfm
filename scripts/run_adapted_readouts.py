#!/usr/bin/env python3
"""Final-layer readouts from existing adapters; never optimize encoder weights.

Matched to connectivity_readouts_v2. Derived scratch features are removed ONLY
after their predictions/similarity matrices have been saved, retaining immutable
checkpoints, input hashes, per-seed predictions, and all ensemble inference.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import time
from functools import lru_cache
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import run_connectivity_readouts as frozen
from fm_matrix.data import save,sha
from fm_readout_v4.evaluate import tag

RUN=ROOT/'runs/adapted_readouts_v1'
OUT=ROOT/'results/generated/adapted_readouts_v1'
FROZEN=frozen.RUN
MODELS=frozen.MODELS
SEEDS=(42,43,44)

def read(p):return json.loads(Path(p).read_text())
def state_id(s):return tag(*s)
def group_id(s):return state_id((*s[:4],0)).removesuffix('_s0')
def directory(s):return RUN/'states'/state_id(s)
def scope(s,co):return 'source_exposed_descriptive' if co==s[0] or (co=='nacc_repeated' and s[0]=='nacc') else 'external'

def states():
    for model in MODELS:
        for source in ('adni','nacc'):
            for objective,target in [('supervised','MEM'),('supervised','EF'),('native_ssl','none')]:
                for seed in SEEDS:yield (source,model,objective,target,seed)

def training_run(s):
    source,model,obj,target,seed=s
    if model=='brainharmonix_f':return ROOT/'runs/brainharmonix_f_v1',model
    if model in ('fmrilm_mae','fmrilm_vq_domain'):
        return ROOT/'runs/fmrilm_variants_v1'/('mae' if model=='fmrilm_mae' else 'vq-domain'),'fmrilm'
    if model in ('semantoks','fmrilm'):return ROOT/'runs/new_fm_v1',model
    if model=='brainlm13' and obj=='supervised':return ROOT/'runs/fm_cls_posttraining_v5',model
    return ROOT/'runs/fm_posttraining_matrix_v2',model

def adapter_path(s):
    source,model,obj,target,seed=s;run,internal=training_run(s)
    return run/'training'/source/internal/f'{obj}_{target}_s{seed}_external'/('fold0/adapter.pt' if obj=='supervised' else 'adapter.pt')

def parent_folder(s):
    from unified_supplements import feature_folder
    if s[1]=='brainharmonix_f':return ROOT/'runs/brainharmonix_f_v1/features'/state_id(s)
    return feature_folder(tuple(s))

def zero_adapter(adapter):
    # A differs at initialization across seeds, but B==0 gives exactly zero update.
    b=[v for k,v in adapter.items() if k.endswith('lora_B')]
    if not b or any(not k.endswith(('lora_A','lora_B')) for k in adapter):raise ValueError('Unexpected adapter tensors')
    return all(not bool(v.count_nonzero()) for v in b)

def init():
    import torch
    records=[];missing=[]
    for s in states():
        ck=adapter_path(s);parent=parent_folder(s);run,internal=training_run(s)
        if not ck.exists() or not (parent/'complete.json').exists():missing.append(state_id(s));continue
        d=torch.load(ck,map_location='cpu',weights_only=True);stamp=read(parent/'complete.json');signature=stamp['signature']
        expected=signature.get('checkpoint_sha256',signature.get('adapter_sha256'))
        if expected!=sha(ck) or d['model']!=internal or d['intake_sha256']!=sha(run/'intake.json'):
            raise ValueError('Adapter provenance mismatch '+state_id(s))
        if s[1]=='brainlm13' and s[2]=='supervised' and d['provenance'].get('training_readout')!='CLS_final':
            raise ValueError('Need current CLS-trained13M, not historical L3-trained adapters')
        for co in frozen.COHORTS:
            if not (parent/(co+'.npy')).exists():missing.append(state_id(s)+'/'+co)
        records.append(dict(state=s,id=state_id(s),group=group_id(s),adapter_path=str(ck),adapter_sha256=sha(ck),
            source_checkpoint_sha256=d['source_checkpoint_sha256'],zero_adapter=zero_adapter(d['adapter']),
            best_epoch=d.get('best_epoch'),parent_complete_sha256=sha(parent/'complete.json')))
    if missing:raise ValueError('Missing registered state/input: '+str(missing))
    d=dict(status='registered',states=records,code_sha256=sha(Path(__file__)),frozen_engine_sha256=sha(Path(frozen.__file__)),
        frozen_manifest_sha256=sha(FROZEN/'manifest.json'),readouts={m:frozen.readout_names(m) for m in MODELS},
        encoder_training=False,layer_policy='All readouts from final normalized encoder output; no layer3 or tokenizer-stage contrasts.',
        scope='Nine configurations; ADNI/NACC MEM/EF/SSL existing external adapters; seeds42/43/44. Two-cohort cognition and seven-cohort identification, matched to frozen readout panel.',
        source_evaluation='Source-exposed encoder probes are descriptive and in separate families; no claim of independent within-source encoder validation. Fold-specific encoder retraining is out of scope.',
        aggregation='Average cognitive OOF predictions across split seeds0/1/2 then adapter seeds42/43/44. Average identity similarity matrices across adapter seeds before scoring; never average embeddings or top1 accuracies.',
        inference='5000 participant bootstraps/20000 conditional swaps. Holm across nine models, both sources, all final readouts and target cohorts, separated by task/metric, identity mode, adaptation objective (MEM,EF,SSL), source exposure and comparison type. Readout-vs-native, adapted-vs-frozen for same readout, and adapted-vsSF100FC.',
        disk_policy='New derived scratch arrays only: delete after their cohort-level predictions and identity matrices are saved and validated. Immutable original checkpoints, feature caches and frozen results are preserved.',
        zero_adapter_policy='Reuse frozen readouts only when all LoRA B tensors are exactly zero and archived primary representations reproduce frozen inputs. No GPU inference or new fitting needed for these states.',
        automatic_shutdown=False)
    dest=RUN/'manifest.json'
    if dest.exists():
        if read(dest)!=json.loads(json.dumps(d)):raise ValueError('Registered adapted experiment changed')
    else:save(dest,d)
    print(json.dumps(dict(states=len(records),zero_adapter_states=sum(r['zero_adapter'] for r in records),groups=len({r['group'] for r in records}))),flush=True)
    return d

@lru_cache(None)
def records():return {r['id']:r for r in read(RUN/'manifest.json')['states']}

def load_encoder(s,device):
    import torch
    model=s[1]
    if model in MODELS[:4]:
        from fm_readout_v4.models import Encoder
        net=Encoder(model,device)
    elif model=='brainharmonix_f':
        from brainharmonix.models import Encoder
        net=Encoder(model,device)
    elif model in ('semantoks','fmrilm'):
        from new_fm.models import Encoder
        net=Encoder(model,device)
    else:
        from fmrilm_variants.models import VariantEncoder
        class Encoder(VariantEncoder):pass
        Encoder.variant='mae' if model=='fmrilm_mae' else 'vq-domain'
        net=Encoder('fmrilm',device)
    record=records()[state_id(s)];ck=adapter_path(s)
    if sha(ck)!=record['adapter_sha256'] or sha(net.checkpoint_path)!=record['source_checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    d=torch.load(ck,map_location='cpu',weights_only=True);net.load_adapter(d['adapter']);net.requires_grad_(False).eval()
    return net

def inputs(model,co):
    if model in MODELS[:4]:
        from fm_readout_v4.data import dataset
        return dataset(co,model)
    if model=='brainharmonix_f':
        from brainharmonix.data import dataset
        return dataset(co,model)
    from new_fm.data import dataset
    return dataset(co,'semantoks' if model=='semantoks' else 'fmrilm')

def last_readouts(net,model,x):
    """Exactly the final-layer operators of the frozen panel, with no layer sweep."""
    import torch
    vals={}
    if model=='brainlm13':
        o=net.net.vit(signal_vectors=x,xyz_vectors=net.xyz[None].expand(len(x),-1,-1),noise=None,return_dict=True)
        native=o.last_hidden_state[:,0];regions=o.last_hidden_state[:,1:].float().reshape(len(x),424,10,-1).mean(2)
        vals['content_mean']=regions.mean(1).cpu().numpy()
    elif model in ('brainlm111','brainlm650'):
        from fm_matched_v3.models import patch_coverage
        z=net.tokens(x);native=z[:,0]
        regions=frozen.spatial_patch_mean(z[:,1:].float().cpu().numpy(),net.cfg.image_size,net.cfg.patch_size)
        w=patch_coverage(424,200,net.cfg.image_size,net.cfg.patch_size)
        vals['content_mean']=np.einsum('bnd,n->bd',z[:,1:].float().cpu().numpy(),w/w.sum()).astype(np.float32)
    elif model=='brainjepa':
        z=net.tokens(x);native=z.mean((1,2));regions=z.float().mean(2)
    elif model=='semantoks':
        z=net.tokens(x);native=z[:,0];observed=z[:,1:].reshape(len(z),9,5,-1)[:,:8]
        # Retain the parent Semantoks reduction dtype before casting cached summaries.
        regions=observed.mean(2)
        vals['content_mean']=observed.mean((1,2)).float().cpu().numpy()
        vals['cls_mean_concat']=torch.cat((observed.mean((1,2)),z[:,0]),dim=1).float().cpu().numpy()
    elif model=='brainharmonix_f':
        z,mask=net.token_outputs(x);z=z.float();native=(z*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        regions=z.reshape(len(x),400,18,768).sum(2)/mask.reshape(len(x),400,18).sum(2)[:,:,None]
        vals['token_std']=torch.sqrt(((z-native[:,None]).square()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).cpu().numpy()
    else:
        z=net.tokens(x);native=z.mean(1);regions=z
    if isinstance(regions,torch.Tensor):regions=regions.float().cpu().numpy()
    vals.update(native=native.float().cpu().numpy(),**frozen.regional_readouts(regions))
    if set(vals)!=set(frozen.readout_names(model)) or not all(np.isfinite(v).all() for v in vals.values()):raise ValueError('Invalid readout extraction')
    return vals

def extract(s,co,net,device,smoke=False):
    import torch
    from fm_readout_v4.data import metadata
    model=s[1];m=metadata(co);x,mm=inputs(model,co)
    if m.scan.tolist()!=mm.scan.tolist():raise ValueError('Scan order changed')
    folder=directory(s)/('smoke' if smoke else 'scratch')/co;folder.mkdir(parents=True,exist_ok=True)
    record=records()[state_id(s)];batch=4 if model=='semantoks' or model.startswith('fmrilm') else 1
    n=min(max(2,batch),len(x)) if smoke else len(x)
    dest=folder/'complete.json'
    if dest.exists():return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in read(dest)['features_sha256']}
    request=dict(adapter_sha256=record['adapter_sha256'],code_sha256=sha(Path(__file__)),device=device,n=n,batch=batch,scan_ids=m.scan.tolist())
    req=folder/'request.json'
    if req.exists() and read(req)!=request:raise ValueError('Partial extraction settings changed')
    if not req.exists():save(req,request)
    progress=folder/'progress.json';start=read(progress)['next'] if progress.exists() else 0
    arrays={}
    if start:
        arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='r+') for k in frozen.readout_names(model)}
    for i in range(start,n,batch):
        end=min(i+batch,n)
        with torch.inference_mode(),torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):
            vals=last_readouts(net,model,x[i:end].to(device))
        if not arrays:
            arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=(n,v.shape[1])) for k,v in vals.items()}
        for k,v in vals.items():arrays[k][i:end]=v
        if i%100==0 or end==n:
            for a in arrays.values():a.flush()
            save(progress,dict(next=end,total=n));print('extract',state_id(s),co,end,n,flush=True)
    expected=np.load(parent_folder(s)/(co+'.npy'),mmap_mode='r')[:n]
    actual=arrays['native'];diff=np.asarray(actual,float)-np.asarray(expected,float)
    rmse=float(np.sqrt(np.mean(diff**2))/max(np.sqrt(np.mean(np.asarray(expected,float)**2)),1e-12))
    if rmse>.005:raise ValueError('Primary adapter feature mismatch: '+str(rmse))
    save(dest,dict(status='complete',state=s,cohort=co,scan_ids=m.scan.tolist(),adapter_sha256=record['adapter_sha256'],
        primary_relative_rmse=rmse,primary_max_abs=float(np.max(abs(diff))),features_sha256={k:sha(folder/(k+'.npy')) for k in arrays},
        scope=scope(s,co),layer='final encoder output after final normalization',smoke_only=smoke))
    return arrays

def feature_set(s,co):
    model=s[1]
    if records()[state_id(s)]['zero_adapter']:
        xs=frozen.features(model,co);xs={k:xs[k] for k in frozen.readout_names(model)}
        parent=np.load(parent_folder(s)/(co+'.npy'),mmap_mode='r')
        np.testing.assert_allclose(xs['native'],parent,rtol=1e-4,atol=1e-4)
        return xs
    folder=directory(s)/'scratch'/co;d=read(folder/'complete.json')
    return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in d['features_sha256']}

def predict(s,co,ep,device):
    import sf100_benchmark as sf
    from benchmark_evaluation import predict_prepared,score
    dest=directory(s)/'cognition'/(co+'_'+ep+'.json')
    if dest.exists():return
    old=read(sf.RUN/'evaluation'/co/(ep+'.json'));m=sf.labels(co)
    ix=m.set_index('scan').index.get_indexer(old['scan_ids']);y=np.asarray(old['y']);g=np.asarray(old['subjects'])
    if (ix<0).any() or m.iloc[ix].subject.astype(str).tolist()!=old['subjects']:raise ValueError('Sample mismatch')
    xs=feature_set(s,co);pp={};byseed={}
    if records()[state_id(s)]['zero_adapter']:
        f=read(FROZEN/'cognition'/s[1]/(co+'_'+ep+'.json'))
        if f['scan_ids']!=old['scan_ids'] or f['subjects']!=old['subjects']:raise ValueError('Frozen sample mismatch')
        for key in frozen.readout_names(s[1]):pp[key]=f['predictions'][key];byseed[key]=f['predictions_by_seed'][key]
    else:
        for key,x in xs.items():
            cache=directory(s)/'prediction_parts'/f'{co}_{ep}_{key}.json'
            if cache.exists():
                d=read(cache);pp[key]=d['prediction'];byseed[key]=d['byseed'];continue
            prepared={v:[] for v in (0,1,2)}
            for f in old['split_audit']:
                tr,te,it,iv=[np.asarray(f[k],int) for k in ('train','test','inner_train','validation')]
                if set(g[tr])&set(g[te]) or set(g[it])&set(g[iv]):raise ValueError('Subject leakage')
                prepared[f['seed']].append((tr,te,frozen.project_indices(x,ix[tr],ix[te],100,device),it,iv,frozen.project_indices(x,ix[it],ix[iv],100,device)))
            byseed[key]=[predict_prepared(prepared[v],y,binary=False) for v in (0,1,2)];pp[key]=np.mean(byseed[key],axis=0)
            save(cache,dict(prediction=pp[key],byseed=byseed[key]));print('predict',state_id(s),co,ep,key,flush=True)
    save(dest,dict(status='complete',state=s,cohort=co,task=ep,scope=scope(s,co),subjects=old['subjects'],scan_ids=old['scan_ids'],y=y,
        predictions=pp,predictions_by_seed=byseed,scores={k:score(y,p,False) for k,p in pp.items()},template_sha256=sha(sf.RUN/'evaluation'/co/(ep+'.json'))))

def identity(s,co):
    import sample_alignment as sa
    import readout_sensitivity as rs
    from unified_supplements import projected_similarity
    out=directory(s)/'identity';dest=out/(co+'.json')
    if dest.exists():return
    subjects,(ca,ia),(cb,ib)=sa.paired_rows(co);a=feature_set(s,ca);b=a if ca==cb else feature_set(s,cb);sim={}
    for mode in ('native','projected128'):
        fn=rs.similarity if mode=='native' else projected_similarity
        for key in frozen.readout_names(s[1]):sim[mode+'__'+key]=fn(a[key][ia],b[key][ib])
    out.mkdir(parents=True,exist_ok=True);path=out/(co+'.npz');tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**sim);tmp.replace(path)
    save(dest,dict(status='complete',state=s,cohort=co,scope=scope(s,co),subjects=subjects,similarity_sha256=sha(path)))

def expected_outputs(s,co):
    root=directory(s);out=[]
    if co in ('adni','nacc'):out += [root/'cognition'/(co+'_'+ep+'.json') for ep in ('MEM','EF')]
    if co in frozen.PAIRS:out += [root/'identity'/(co+'.json')]
    if co in ('hrv_pre','hrv_post'):out += [root/'identity/hrv.json']
    return out

def cleanup(s,co):
    # Only files created in this explicitly declared scratch directory are eligible.
    folder=directory(s)/'scratch'/co
    if not folder.exists() or not all(p.exists() for p in expected_outputs(s,co)):return
    if not folder.resolve().is_relative_to((RUN/'states').resolve()):raise ValueError('Scratch path escapes new run')
    for p in folder.glob('*.npy'):
        if p.is_symlink():raise ValueError('Refuse to remove linked parent features')
        p.unlink()
    save(folder/'evicted.json',dict(reason='All dependent predictions and identity matrices saved; original checkpoints and feature caches retained.',utc=time.time()))

def worker(s,device,smoke=False):
    import torch
    torch.set_num_threads(2);root=directory(s);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'complete.json').exists() and not smoke:return
    zero=records()[state_id(s)]['zero_adapter'];net=None if zero else load_encoder(s,device)
    if smoke:
        if zero:feature_set(s,'psopt');d=dict(status='passed',zero_adapter=True)
        else:
            extract(s,'psopt',net,device,smoke=True);d=read(root/'smoke/psopt/complete.json')
            # Verify every operator against the frozen panel using the same input
            # and restored zero-update encoder, without modifying saved adapters.
            with torch.no_grad():
                for name,param in net.named_parameters():
                    if name.endswith('lora_B'):param.zero_()
            x,_=inputs(s[1],'psopt');n=4 if s[1]=='semantoks' or s[1].startswith('fmrilm') else 2
            vals=[]
            batch=4 if n==4 else 1
            with torch.inference_mode(),torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):
                for i in range(0,n,batch):vals.append(last_readouts(net,s[1],x[i:i+batch].to(device)))
            expected=frozen.features(s[1],'psopt')
            for key in frozen.readout_names(s[1]):
                np.testing.assert_allclose(np.concatenate([v[key] for v in vals]),expected[key][:n],rtol=1e-4,atol=1e-4)
            d['all_readouts_reproduce_frozen_operators']=True
        save(root/'smoke.json',d);print(state_id(s),d.get('primary_relative_rmse',0),flush=True);return
    for co in frozen.COHORTS:
        if all(p.exists() for p in expected_outputs(s,co)):
            cleanup(s,co);continue
        if not zero:extract(s,co,net,device)
        if co in ('adni','nacc'):
            for ep in ('MEM','EF'):predict(s,co,ep,device)
        if co in frozen.PAIRS:identity(s,co)
        if co=='hrv_post':identity(s,'hrv');cleanup(s,'hrv_pre')
        cleanup(s,co)
    if not all(p.exists() for co in frozen.COHORTS for p in expected_outputs(s,co)):raise ValueError('Incomplete state outputs')
    save(root/'complete.json',dict(status='complete',state=s,zero_adapter=zero,adapter_sha256=records()[state_id(s)]['adapter_sha256'],new_training=False))

def aligned_ensemble(rows,field):
    if len(rows)!=3 or sorted(int(r['state'][4]) for r in rows)!=list(SEEDS):raise ValueError('Need exactly three adapter seeds')
    first=rows[0]
    for r in rows[1:]:
        if list(first['state'][:4])!=list(r['state'][:4]):raise ValueError('Different adaptation states')
        for key in ('subjects','scan_ids','y'):
            if key in first and first[key]!=r[key]:raise ValueError('Unaligned '+key)
        if set(first[field])!=set(r[field]):raise ValueError('Readout keys differ')
    return {k:np.mean([r[field][k] for r in rows],axis=0) for k in first[field]}

def inference(s,co,ep=None):
    from paper_paired_stats import paired
    import readout_sensitivity as rs
    import sf100_benchmark as sf
    import sample_alignment as sa
    from unified_supplements import projected_similarity
    model=s[1];ss=[(*s[:4],seed) for seed in SEEDS];group=group_id(s)
    out=RUN/'inference'/group/(co+('_'+ep if ep else '')+'.json')
    if out.exists():return
    rows=[];scores=[];exposure=scope(s,co);objective=s[3] if s[2]=='supervised' else 'SSL'
    def info(typ,metric,mode=''):
        return dict(model=model,source=s[0],objective=objective,scope=exposure,cohort=co,task=ep or 'identification',metric=metric,mode=mode,
                    family='adapted_readouts::'+'::'.join([typ,ep or 'identification',metric,mode,objective,exposure]))
    if ep:
        rr=[read(directory(t)/'cognition'/(co+'_'+ep+'.json')) for t in ss];p=aligned_ensemble(rr,'predictions')
        fr=read(FROZEN/'cognition'/model/(co+'_'+ep+'.json'))
        if fr['subjects']!=rr[0]['subjects'] or fr['scan_ids']!=rr[0]['scan_ids'] or fr['y']!=rr[0]['y']:raise ValueError('Frozen alignment failed')
        y=rr[0]['y'];groups=rr[0]['subjects'];fixed=fr['predictions'];comp=[]
        for key in frozen.readout_names(model):
            comp.extend([('adapted_vs_frozen',key,key,fixed[key],p[key]),('vs_sf100','sf100_fc',key,fixed['sf100_fc'],p[key])])
            if key!='native':comp.append(('readout_vs_native','native',key,p['native'],p[key]))
        for typ,ref,cand,a,b in comp:
            st=paired(y,a,b,groups,seed=frozen.seed(group+co+ep+typ+cand))
            rows.append({**info(typ,'rho'),**st,'reference':ref,'candidate':cand})
        scores=[dict(readout=k,score=float(rs.score(y,v,False))) for k,v in p.items()]
        save(out,dict(status='complete',rows=rows,scores=scores,ensemble_predictions=p,subjects=groups,y=y,scan_ids=rr[0]['scan_ids']))
    else:
        rr=[]
        for t in ss:
            r=read(directory(t)/'identity'/(co+'.json'));path=directory(t)/'identity'/(co+'.npz')
            if sha(path)!=r['similarity_sha256']:raise ValueError('Similarity changed')
            with np.load(path) as z:r['sim']={k:z[k] for k in z.files}
            rr.append(r)
        sims=aligned_ensemble(rr,'sim');fr=read(FROZEN/'identity'/model/(co+'.json'))
        if fr['subjects']!=rr[0]['subjects']:raise ValueError('Frozen identity sample mismatch')
        for mode in ('native','projected128'):
            adapted={k:rs.identification_scores(sims[mode+'__'+k]) for k in frozen.readout_names(model)}
            fixed=fr['scores'][mode]
            for key,b in adapted.items():
                for metric in ('top1','I_diff'):
                    scores.append(dict(readout=key,mode=mode,metric=metric,score=b[metric]))
                    comparisons=[('adapted_vs_frozen',key,fixed[key]),('vs_sf100','sf100_fc',fixed['sf100_fc'])]
                    if key!='native':comparisons.append(('readout_vs_native','native',adapted['native']))
                    for typ,ref,a in comparisons:
                        st=rs.paired_identification(a['per_subject_'+metric],b['per_subject_'+metric],frozen.seed(group+co+mode+metric+typ+key))
                        rows.append({**info(typ,metric,mode),**st,'reference':ref,'candidate':key})
        save(out,dict(status='complete',rows=rows,scores=scores,subjects=rr[0]['subjects'],aggregation='Average similarities before scoring'))

def jobs(d):
    result=[]
    for r in d['states']:
        result.append(dict(id=r['id'],action='worker',state=r['state'],deps=[],cpu=r['zero_adapter'],output=str(directory(r['state'])/'complete.json')))
    for r in d['states']:
        s=r['state']
        if s[4]!=42:continue
        deps=[state_id((*s[:4],seed)) for seed in SEEDS]
        for co in ('adni','nacc'):
            for ep in ('MEM','EF'):result.append(dict(id=r['group']+'/'+co+'/'+ep,action='inference',state=s,cohort=co,endpoint=ep,deps=deps,cpu=True,output=str(RUN/'inference'/r['group']/(co+'_'+ep+'.json'))))
        for co in frozen.PAIRS:result.append(dict(id=r['group']+'/'+co+'/identity',action='inference',state=s,cohort=co,deps=deps,cpu=True,output=str(RUN/'inference'/r['group']/(co+'.json'))))
    return result

def report(d):
    from paper_paired_stats import holm
    alljobs=jobs(d);coverage=[];rows=[];scores=[]
    for j in alljobs:
        p=Path(j['output']);done=p.exists();coverage.append(dict(id=j['id'],status='complete' if done else 'pending'))
        if done and j['action']=='inference':
            r=read(p);rows+=r['rows']
            scores += [dict(model=j['state'][1],source=j['state'][0],objective=j['state'][3] if j['state'][2]=='supervised' else 'SSL',
                cohort=j['cohort'],task=j.get('endpoint','identification'),scope=scope(j['state'],j['cohort']),**v) for v in r['scores']]
    complete=all(c['status']=='complete' for c in coverage);OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(coverage).to_csv(OUT/'Coverage.csv',index=False)
    if rows:
        df=pd.DataFrame(rows);df['holm_p']=np.nan
        if complete:
            for _,ix in df.groupby('family').groups.items():df.loc[ix,'holm_p']=holm(df.loc[ix,'permutation_p'].fillna(1).tolist())
        df['inference_status']='final' if complete else 'pending_complete_family'
        df.to_csv(OUT/'Comparisons.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'Scores.csv',index=False)
        with pd.ExcelWriter(OUT/'Posttrained_readouts.xlsx',engine='xlsxwriter') as w:
            df.to_excel(w,sheet_name='Comparisons',index=False);pd.DataFrame(scores).to_excel(w,sheet_name='Scores',index=False)
            pd.DataFrame(coverage).to_excel(w,sheet_name='Coverage',index=False)
    result=dict(total=len(alljobs),complete=sum(c['status']=='complete' for c in coverage),all_complete=complete)
    save(RUN/'summary.json',result)
    if complete:save(RUN/'complete.json',result)
    return result

def run(device,workers):
    import torch
    d=read(RUN/'manifest.json')
    if d['code_sha256']!=sha(Path(__file__)) or d['frozen_engine_sha256']!=sha(Path(frozen.__file__)):raise ValueError('Registered implementation changed')
    if not torch.cuda.is_available():raise RuntimeError('GPU not available; no silent CPU inference')
    lock=(RUN/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    alljobs=jobs(d);active={};failed=set();logs=ROOT/'logs/adapted_readouts_v1';logs.mkdir(parents=True,exist_ok=True)
    while True:
        for key,(proc,j,fh) in list(active.items()):
            if proc.poll() is None:continue
            fh.close();del active[key]
            if proc.returncode or not Path(j['output']).exists():failed.add(key)
            print('finished',key,'exit',proc.returncode,flush=True)
        done={j['id'] for j in alljobs if Path(j['output']).exists()}
        for j in sorted(alljobs,key=lambda j:j['action']=='worker'):
            if j['id'] in done|set(active)|failed or not set(j['deps'])<=done:continue
            if sum(v[1]['cpu']==j['cpu'] for v in active.values())>=workers:continue
            if shutil.disk_usage(RUN).free<30*1024**3:break
            cmd=[sys.executable,str(Path(__file__).resolve()),j['action'],'--state',state_id(j['state']),'--device','cpu' if j['cpu'] else device]
            if 'cohort' in j:cmd+=['--cohort',j['cohort']]
            if 'endpoint' in j:cmd+=['--endpoint',j['endpoint']]
            fh=open(logs/(j['id'].replace('/','_')+'.log'),'a');active[j['id']]=(subprocess.Popen(cmd,stdout=fh,stderr=subprocess.STDOUT),j,fh)
            print('started',j['id'],flush=True)
        save(RUN/'status.json',dict(completed=len(done),total=len(alljobs),running=list(active),failed=sorted(failed),pending=len(alljobs)-len(done)-len(active),
            gpu_available=True,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),automatic_shutdown=False))
        if not active:break
        time.sleep(10)
    print(report(d),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','smoke','worker','inference','run','report','validate']);p.add_argument('--state');p.add_argument('--device',default='cuda');p.add_argument('--cohort');p.add_argument('--endpoint');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.action=='init':init();return
    if a.action=='run':run(a.device,a.workers);return
    if a.action=='report':print(report(read(RUN/'manifest.json')));return
    if a.action=='validate':
        checked=[]
        for model in MODELS:
            objective='native_ssl' if model in ('brainjepa','semantoks','brainharmonix_f') else 'supervised'
            r=next(r for r in records().values() if r['state'][1]==model and r['state'][2]==objective and not r['zero_adapter'])
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'smoke','--state',r['id'],'--device',a.device],check=True);checked.append(r['id'])
        save(RUN/'validation.json',dict(status='passed',states=checked,code_sha256=sha(Path(__file__))))
        return
    s=tuple(records()[a.state]['state'])
    if a.action in ('worker','smoke'):worker(s,a.device,a.action=='smoke')
    else:inference(s,a.cohort,a.endpoint)

if __name__=='__main__':main()
