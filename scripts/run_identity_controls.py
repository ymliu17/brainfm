#!/usr/bin/env python3
"""BrainLM13 final-layer identification controls. Additive, resumable, no shutdown."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
from pathlib import Path
import argparse, copy, fcntl, hashlib, json, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from fm_matrix.data import save,sha
import run_connectivity_readouts as frozen
import readout_sensitivity as rs
from fm_readout_v4.data import metadata,dataset
from unified_supplements import projected_similarity
from paper_paired_stats import holm
RUN=ROOT/'runs/brainlm13_identity_controls_v1'
OUT=ROOT/'results/generated/brainlm13_identity_controls_v1'
COHORTS=('adni','nacc_repeated','act','cogte','breathe','psopt','hrv')
SEEDS=(101,102,103,104,105)
MODES=('native','projected128')
METRICS=('top1','I_diff')
FD=ROOT/'data/rct/psopt/pSOPT_adsct_fc_meanFD.csv'

def read(p):return json.loads(Path(p).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def paired(co):return read(RUN/'pairs'/f'{co}.json')
def target(state,co):return RUN/'features'/state/co

def register():
    import sample_alignment as sa
    RUN.mkdir(parents=True,exist_ok=True)
    pairs={};sources={}
    for co in COHORTS:
        subjects,(ca,ia),(cb,ib)=sa.paired_rows(co)
        old=read(frozen.RUN/'identity/brainlm13'/f'{co}.json')
        assert subjects==old['subjects']
        record=dict(subjects=subjects,a=dict(cohort=ca,indices=ia,scans=metadata(ca).iloc[ia].scan.tolist()),
                    b=dict(cohort=cb,indices=ib,scans=metadata(cb).iloc[ib].scan.tolist()))
        p=RUN/'pairs'/f'{co}.json'
        from fm_matrix.data import clean
        if p.exists():assert read(p)==clean(record)
        else:save(p,record)
        pairs[co]=len(subjects)
        sources[str(p.relative_to(ROOT))]=sha(p)
    protocol=dict(version=1,cohort_pairs=pairs,random_seeds=SEEDS,
        scope='BrainLM13 only, frozen pretrained versus five freshly initialized encoders. Seven matched cross-visit cohorts. Supplementary retrospective controls; no cognitive models or post-training.',
        readouts='Final normalized layer only: CLS and connectivity of time-averaged ROI embeddings. No layer/readout selection.',
        random='Fresh BrainLMForPretraining constructor with identical config and coordinates, default initializer, no checkpoint weights loaded into random models. One fixed seed/encoder across all cohorts and visits. No adapters or fitting.',
        extraction='Same canonical A424200 input, bfloat16 autocast, batch1, eval/unmasked. Pretrained cache reproduced on two paired scans per cohort before random extraction.',
        summaries='ROI standard deviation (ddof0) and lag1 autocorrelation from retained real input frames only. No mean signal control because input is ROI-centered. No claim of tissue-specific or neural origin.',
        FC='Reuse hash-verified A424 same-window Fisher-z FC and parent identification scores. Native dimensions plus same fixed Gaussian128 projection seeds0/1/2.',
        statistics='Bidirectional fractional-tie top1 and I_diff. 5000 participant bootstraps,20000 two-sided paired query-score swaps conditional on fixed gallery. Random reference is seed-average per-participant score, not an ensemble of encoders. All seed-specific scores retained.',
        primary_contrasts='Pretrained vs seed-average random at each readout; connectivity-vs-CLS gain difference (pretrained minus seed-average random); pretrained connectivity vs FC/std/lag1. Holm over all seven cohorts and contrasts, separately by mode and metric. No new absolute-null claims.',
        motion='pSOPT T1/T2 mean FD only where both match. Motion-only similarity = negative absolute log1pFD difference. OLS feature residualization on log1pFD learned from20% participant calibration set, applied to disjoint80% identification gallery/query; seeds0/1/2 separately, adjusted and unadjusted same gallery. Holm across all split/model contrasts separately by mode/metric. Does not remove all motion confounding.',
        shutdown=False,manuscript_modified=False,code_sha256=sha(Path(__file__)),pair_sources=sources,
        helper_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(rs.__file__),ROOT/'src/unified_supplements.py',ROOT/'src/model_loader.py',ROOT/'src/brainlm_mae/modeling_brainlm.py']},
        motion_source_sha256=sha(FD))
    from fm_matrix.data import clean
    p=RUN/'protocol.json'
    if p.exists():assert read(p)==clean(protocol),'Registered protocol changed'
    else:save(p,protocol)
    print(json.dumps(pairs),flush=True)

def verify():
    p=read(RUN/'protocol.json');assert p['code_sha256']==sha(Path(__file__))
    for name,h in {**p['helper_sha256'],**p['pair_sources']}.items():assert sha(ROOT/name)==h,name
    assert sha(FD)==p['motion_source_sha256']

def array_save(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**arrays);tmp.replace(path)

def load_arrays(state,co):
    path=target(state,co)/'features.npz';st=read(path.parent/'complete.json')
    assert st['sha256']==sha(path)
    assert st['pair_sha256']==sha(RUN/'pairs'/f'{co}.json')
    with np.load(path) as z:return {k:z[k] for k in z.files}

def finish_arrays(state,co,arrays,**extra):
    folder=target(state,co);p=folder/'features.npz';array_save(p,**arrays)
    save(folder/'complete.json',dict(status='complete',state=state,cohort=co,sha256=sha(p),
        pair_sha256=sha(RUN/'pairs'/f'{co}.json'),shapes={k:list(v.shape) for k,v in arrays.items()},**extra))

def inputs(co):
    p=paired(co);out=[];ms=[]
    for side in ('a','b'):
        v=p[side];x,m=dataset(v['cohort'],'brainlm13');idx=v['indices']
        assert m.iloc[idx].scan.tolist()==v['scans']
        out.append(x[idx]);ms.append(m.iloc[idx].reset_index(drop=True))
    return out,ms

def input_summaries(x,n):
    x=np.asarray(x[:,:n],dtype=float)
    sd=x.std(1)
    a=x[:,:-1]-x[:,:-1].mean(1,keepdims=True);b=x[:,1:]-x[:,1:].mean(1,keepdims=True)
    denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    lag=np.divide((a*b).sum(1),denom,out=np.zeros(len(x)),where=denom>1e-12)
    return sd.astype('float32'),lag.astype('float32')

def controls(co):
    folder=target('controls',co)
    if (folder/'complete.json').exists():load_arrays('controls',co);return
    p=paired(co);xx,mm=inputs(co);arrays={};sources={}
    for side,x,m in zip(('a','b'),xx,mm):
        v=p[side];root=frozen.RUN/'features/brainlm13'/v['cohort'];stamp=read(root/'complete.json')
        assert stamp['scan_ids']==metadata(v['cohort']).scan.tolist()
        for name in ('native','connectivity'):
            path=root/(name+'.npy');assert sha(path)==stamp['features_sha256'][name]
            arrays[f'pretrained_{name}_{side}']=np.load(path,mmap_mode='r')[v['indices']].copy();sources[str(path)]=sha(path)
        sd=[];lag=[]
        for a,r in zip(x.numpy(),m.itertuples()):
            n=min(int(r.native_frames),200);assert n>=3
            s,l=input_summaries(a,n);sd.append(s);lag.append(l)
        arrays[f'roi_std_{side}']=np.array(sd);arrays[f'roi_lag1_{side}']=np.array(lag)
        path=ROOT/'runs/reviewer_completion_v1/input_fc/a424'/(v['cohort']+'.npy');stamp=read(path.with_suffix('.json'))
        assert stamp['scan_ids']==metadata(v['cohort']).scan.tolist() and sha(path)==stamp['sha256']
        arrays[f'FC_{side}']=np.load(path,mmap_mode='r')[v['indices']].copy();sources[str(path)]=sha(path)
    finish_arrays('controls',co,arrays,sources_sha256=sources)
    print('controls complete',co,len(p['subjects']),flush=True)

def create_encoder(seed=None):
    import torch,config as C
    from paths import OFFICIAL_BRAINLM_13M,XYZ_PATH
    from datasets import load_xyz
    from model_loader import load_brainlm_forpretraining
    from brainlm_mae.configuration_brainlm import BrainLMConfig
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    C.NUM_TIMEPOINTS_PER_VOXEL=200
    if seed is None:net=load_brainlm_forpretraining(OFFICIAL_BRAINLM_13M)
    else:
        cfg=BrainLMConfig(hidden_size=C.HIDDEN_SIZE,timepoint_patching_size=C.TIMEPOINT_PATCHING_SIZE,
            num_hidden_layers=C.NUM_HIDDEN_LAYERS,num_attention_heads=C.NUM_ATTENTION_HEADS,intermediate_size=C.INTERMEDIATE_SIZE,
            num_brain_voxels=C.NUM_BRAIN_VOXELS,num_timepoints_per_voxel=200,decoder_hidden_size=C.DECODER_HIDDEN_SIZE,
            decoder_num_hidden_layers=C.DECODER_NUM_HIDDEN_LAYERS,decoder_num_attention_heads=C.DECODER_NUM_ATTENTION_HEADS,
            decoder_intermediate_size=C.DECODER_INTERMEDIATE_SIZE,mask_ratio=C.MASK_RATIO)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);net=BrainLMForPretraining(cfg)
    net.eval().requires_grad_(False)
    return net,load_xyz(XYZ_PATH)

def encoder_hash(net):
    h=hashlib.sha256()
    for name,v in net.vit.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def extract_one(net,xyz,x,device):
    import torch
    with torch.inference_mode(),torch.autocast(device_type=device.split(':')[0],dtype=torch.bfloat16):
        o=net.vit(signal_vectors=x[None].to(device),xyz_vectors=xyz[None],noise=None,return_dict=True)
    z=o.last_hidden_state.float()
    native=z[:,0].cpu().numpy()[0]
    regions=z[:,1:].reshape(1,424,10,-1).mean(2).cpu().numpy()
    return dict(native=native,connectivity=frozen.connectivity(regions)[0])

def smoke(device):
    import torch
    torch.set_num_threads(2)
    net,xyz=create_encoder();net.to(device);xyz=xyz.to(device)
    # Confirm same-pass parity for both visits before trusting the random model comparison.
    errors={}
    for co in COHORTS:
        pairs=paired(co);xx,_=inputs(co)
        for side,x in zip(('a','b'),xx):
            v=pairs[side];z=extract_one(net,xyz,x[0],device)
            for name,f in z.items():
                old=np.load(frozen.RUN/'features/brainlm13'/v['cohort']/(name+'.npy'),mmap_mode='r')[v['indices'][0]]
                errors[f'{co}_{side}_{name}']=float(np.max(np.abs(old-f)))
                np.testing.assert_allclose(f,old,rtol=1e-3,atol=1e-3)
    h=encoder_hash(net);cfg=net.config.to_dict();del net
    a,coords=create_encoder(SEEDS[0]);b,_=create_encoder(SEEDS[0]);c,_=create_encoder(SEEDS[1])
    assert a.config.to_dict()==cfg and encoder_hash(a)==encoder_hash(b)!=encoder_hash(c)
    assert encoder_hash(a)!=h
    a.to(device);coords=coords.to(device)
    z=extract_one(a,coords,xx[0][0],device);z2=extract_one(a,coords,xx[0][0],device)
    for k in z:np.testing.assert_array_equal(z[k],z2[k]);assert np.isfinite(z[k]).all()
    save(RUN/'smoke.json',dict(status='complete',device=device,cache_max_abs_error=errors,
        random_hash=encoder_hash(a),pretrained_hash=h,deterministic=True,config_matched=True,readout_dims={k:len(v) for k,v in z.items()}))
    print('smoke passed',errors,flush=True)

def random_extract(seed,co,device):
    import torch
    state=f'random{seed}';folder=target(state,co)
    if (folder/'complete.json').exists():load_arrays(state,co);return
    torch.set_num_threads(2);net,xyz=create_encoder(seed);h=encoder_hash(net);net.to(device);xyz=xyz.to(device)
    xx,_=inputs(co);arrays={};start=time.time()
    for side,x in zip(('a','b'),xx):
        vals={k:[] for k in ('native','connectivity')}
        for i,row in enumerate(x):
            chunk=folder/'chunks'/f'{side}_{i:04d}.npz'
            if chunk.exists():
                with np.load(chunk) as z:result={k:z[k] for k in z.files}
            else:result=extract_one(net,xyz,row,device);array_save(chunk,**result)
            for k in vals:vals[k].append(result[k])
            if i%50==0:
                save(RUN/'gpu_progress.json',dict(state=state,cohort=co,side=side,done=i+1,total=len(x),elapsed_seconds=time.time()-start,updated=utc()))
                print(state,co,side,i+1,len(x),flush=True)
        arrays.update({f'{k}_{side}':np.stack(v) for k,v in vals.items()})
    finish_arrays(state,co,arrays,encoder_sha256=h,seed=seed,elapsed_seconds=time.time()-start,device=device)
    # Temporary per-scan chunks only; keep final feature arrays and provenance.
    import shutil
    shutil.rmtree(folder/'chunks')

def score_arrays(arrays):
    names=sorted({k[:-2] for k in arrays if k.endswith('_a')});results={}
    for mode in MODES:
        fn=rs.similarity if mode=='native' else projected_similarity
        results[mode]={name:rs.identification_scores(fn(arrays[name+'_a'],arrays[name+'_b'])) for name in names}
    return results

def compare(ref,cand,co,mode,metric,kind,name):
    key=f'{co}|{mode}|{metric}|{kind}|{name}'
    st=rs.paired_identification(ref,cand,seed=frozen.seed(key))
    return dict(cohort=co,mode=mode,metric=metric,kind=kind,comparison=name,family=f'{kind}::{mode}::{metric}',**st)

def evaluate(co):
    dest=RUN/'evaluation'/f'{co}.json'
    if dest.exists():return
    p=paired(co);arrays=load_arrays('controls',co);scores=score_arrays(arrays)
    old=read(frozen.RUN/'identity/brainlm13'/f'{co}.json')
    oldfc=read(ROOT/'runs/reviewer_completion_v1/identity/brainlm13'/f'{co}.json')
    for mode in MODES:
        for name in ('native','connectivity'):
            for met in METRICS:np.testing.assert_allclose(scores[mode]['pretrained_'+name]['per_subject_'+met],old['scores'][mode][name]['per_subject_'+met],atol=1e-10)
        fc=next(x['FC'] for x in oldfc['scores'] if x['mode']==mode)
        for met in METRICS:np.testing.assert_allclose(scores[mode]['FC']['per_subject_'+met],fc['per_subject_'+met],atol=1e-10)
    random={}
    for seed in SEEDS:
        random[str(seed)]=score_arrays(load_arrays(f'random{seed}',co))
    rows=[]
    for mode in MODES:
        for metric in METRICS:
            key='per_subject_'+metric
            pc=np.asarray(scores[mode]['pretrained_connectivity'][key]);pn=np.asarray(scores[mode]['pretrained_native'][key])
            rc=np.mean([random[str(s)][mode]['connectivity'][key] for s in SEEDS],axis=0)
            rn=np.mean([random[str(s)][mode]['native'][key] for s in SEEDS],axis=0)
            for ref,cand,name in [(rn,pn,'pretrained_minus_random_CLS'),(rc,pc,'pretrained_minus_random_connectivity'),
                (rc-rn,pc-pn,'pretraining_by_readout_interaction'),(pn,pc,'pretrained_connectivity_minus_CLS'),(rn,rc,'random_connectivity_minus_CLS')]:
                rows.append(compare(ref,cand,co,mode,metric,'core',name))
            for reference in ('FC','roi_std','roi_lag1'):
                rows.append(compare(scores[mode][reference][key],pc,co,mode,metric,'core','pretrained_connectivity_minus_'+reference))
    save(dest,dict(status='complete',subjects=p['subjects'],scores=scores,random_by_seed=random,rows=rows,parent_scores_reproduced=True))
    print('evaluation complete',co,flush=True)

def motion():
    dest=RUN/'motion.json'
    if dest.exists():return
    from sklearn.model_selection import train_test_split
    co='psopt';p=paired(co);fd=pd.read_csv(FD);assert not fd.duplicated(['subject','timepoint']).any()
    lookup=fd.set_index(['subject','timepoint']).mean_fd
    vals={}
    for side in ('a','b'):
        m=metadata(p[side]['cohort']).iloc[p[side]['indices']]
        vals[side]=np.array([lookup.get((str(s).removeprefix('sub-'),v),np.nan) for s,v in zip(m.subject,m.visit)],float)
    keep=np.isfinite(vals['a'])&np.isfinite(vals['b']);assert (vals['a'][keep]>=0).all() and (vals['b'][keep]>=0).all()
    fd_a,fd_b=[np.log1p(vals[s][keep]) for s in ('a','b')];n=int(keep.sum());assert n>=15
    arrays={k:v[keep] for k,v in load_arrays('controls',co).items()}
    for seed in SEEDS:
        arrays.update({f'random{seed}_{k}':v[keep] for k,v in load_arrays(f'random{seed}',co).items()})
    rows=[];splits=[];motion_only=rs.identification_scores(-np.abs(fd_a[:,None]-fd_b[None,:]))
    for split_seed in (0,1,2):
        cal,ev=train_test_split(np.arange(n),train_size=.2,random_state=split_seed)
        xcal=np.column_stack((np.ones(2*len(cal)),np.r_[fd_a[cal],fd_b[cal]]))
        xa=np.column_stack((np.ones(len(ev)),fd_a[ev]));xb=np.column_stack((np.ones(len(ev)),fd_b[ev]))
        adj={};unadj={}
        for name in sorted({k[:-2] for k in arrays if k.endswith('_a')}):
            a,b=arrays[name+'_a'],arrays[name+'_b']
            beta=np.linalg.lstsq(xcal,np.concatenate([a[cal],b[cal]]),rcond=None)[0]
            adj[name+'_a']=a[ev]-xa@beta;adj[name+'_b']=b[ev]-xb@beta
            unadj[name+'_a']=a[ev];unadj[name+'_b']=b[ev]
        adjusted=score_arrays(adj);original=score_arrays(unadj)
        for mode in MODES:
            for metric in METRICS:
                key='per_subject_'+metric;pre=np.asarray(adjusted[mode]['pretrained_connectivity'][key])
                random_mean=np.mean([adjusted[mode][f'random{s}_connectivity'][key] for s in SEEDS],axis=0)
                for ref,name in [(adjusted[mode]['pretrained_native'][key],'connectivity_minus_CLS'),(adjusted[mode]['FC'][key],'connectivity_minus_FC'),(random_mean,'connectivity_minus_random')]:
                    r=compare(ref,pre,co,mode,metric,'motion',f'split{split_seed}_{name}');r['split_seed']=split_seed;rows.append(r)
                r=compare(original[mode]['pretrained_connectivity'][key],pre,co,mode,metric,'motion',f'split{split_seed}_adjusted_minus_unadjusted');r['split_seed']=split_seed;rows.append(r)
        splits.append(dict(seed=split_seed,calibration_subjects=np.array(p['subjects'])[keep][cal],evaluation_subjects=np.array(p['subjects'])[keep][ev],adjusted=adjusted,unadjusted=original))
    save(dest,dict(status='complete',cohort=co,n=n,subjects=np.array(p['subjects'])[keep],motion_only=motion_only,rows=rows,splits=splits,
        coverage={c:('evaluated' if c=='psopt' else 'no verified paired motion source in this panel') for c in COHORTS},
        source_sha256=sha(FD),note='Motion similarity uses scalar log1p meanFD distance, not feature-wise Pearson correlation. Residualization fits calibration participants only; evaluated participants excluded. Separate fixed-gallery splits.'))

def report():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];scores=[];coverage=[]
    queue=[]
    for co in COHORTS:
        for state in ['controls']+[f'random{s}' for s in SEEDS]:
            path=target(state,co)/'complete.json'
            queue.append(dict(id=state+'/'+co,lane='cpu' if state=='controls' else 'gpu',status='complete' if path.exists() else 'pending',output=str(path.relative_to(ROOT))))
        path=RUN/'evaluation'/f'{co}.json'
        queue.append(dict(id='evaluate/'+co,lane='finish',status='complete' if path.exists() else 'pending',output=str(path.relative_to(ROOT))))
    queue.append(dict(id='motion/psopt',lane='finish',status='complete' if (RUN/'motion.json').exists() else 'pending',output=str((RUN/'motion.json').relative_to(ROOT))))
    save(RUN/'queue_status.json',dict(updated=utc(),tasks=queue,complete=sum(t['status']=='complete' for t in queue),total=len(queue)))
    for co in COHORTS:
        path=RUN/'evaluation'/f'{co}.json';coverage.append(dict(cohort=co,status='complete' if path.exists() else 'pending'))
        if not path.exists():continue
        d=read(path);rows+=d['rows']
        for mode,ss in d['scores'].items():
            for method,s in ss.items():scores.append(dict(cohort=co,mode=mode,method=method,seed=None,**{k:s[k] for k in ('n','top1','I_diff','chance')}))
        for seed,mm in d['random_by_seed'].items():
            for mode,ss in mm.items():
                for method,s in ss.items():scores.append(dict(cohort=co,mode=mode,method='random_'+method,seed=int(seed),**{k:s[k] for k in ('n','top1','I_diff','chance')}))
    done=all(x['status']=='complete' for x in coverage)
    if (RUN/'motion.json').exists():rows+=read(RUN/'motion.json')['rows']
    table=pd.DataFrame(rows)
    if len(table):
        table['holm_p']=np.nan
        for fam,idx in table.groupby('family').groups.items():
            if done if fam.startswith('core::') else (RUN/'motion.json').exists():table.loc[idx,'holm_p']=holm(table.loc[idx,'permutation_p'])
        table.to_csv(OUT/'Comparisons.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'Scores.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'Coverage.csv',index=False)
    save(OUT/'status.json',dict(status='complete' if done and (RUN/'motion.json').exists() else 'in_progress',updated=utc(),coverage=coverage,
        motion_complete=(RUN/'motion.json').exists(),no_parent_results_changed=True))
    if done and (RUN/'motion.json').exists():
        with pd.ExcelWriter(OUT/'Identity_controls.xlsx',engine='xlsxwriter') as w:
            for name,df in [('Scores',pd.DataFrame(scores)),('Paired comparisons',table),('Coverage',pd.DataFrame(coverage))]:df.to_excel(w,sheet_name=name,index=False)
            pd.DataFrame([{'item':k,'definition':str(v)} for k,v in read(RUN/'protocol.json').items()]).to_excel(w,sheet_name='Protocol',index=False)


def worker(lane,device):
    verify();lock=(RUN/f'{lane}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        save(RUN/f'{lane}_status.json',dict(status='running',pid=os.getpid(),updated=utc()))
        if lane=='gpu':
            if not (RUN/'smoke.json').exists():smoke(device)
            for seed in SEEDS:
                for co in COHORTS:random_extract(seed,co,device)
        elif lane=='cpu':
            for co in COHORTS:controls(co)
        else:
            pending=list(COHORTS)
            while pending:
                report()
                for co in pending.copy():
                    if all((target(s,co)/'complete.json').exists() for s in ['controls']+[f'random{s}' for s in SEEDS]):
                        evaluate(co);pending.remove(co);report()
                for other in ('cpu','gpu'):
                    path=RUN/f'{other}_status.json'
                    if path.exists() and read(path)['status']=='failed':raise RuntimeError(f'{other} lane failed; see log')
                if pending:time.sleep(20)
            motion();report()
        save(RUN/f'{lane}_status.json',dict(status='complete',pid=os.getpid(),updated=utc()))
    except Exception as e:
        save(RUN/f'{lane}_status.json',dict(status='failed',pid=os.getpid(),updated=utc(),error=repr(e)));raise


def test():
    x=np.array([[1.,2.,3.,4.],[1.,1.,1.,1.]])
    s,l=input_summaries(x,4);np.testing.assert_allclose(s,[np.sqrt(1.25),0]);np.testing.assert_allclose(l,[1,0])
    a=rs.identification_scores(np.eye(5));assert a['top1']==1
    b=rs.identification_scores(np.ones((5,5)));assert b['top1']==.2
    print('summary and identification checks passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','test','smoke','worker','report']);p.add_argument('--lane',choices=['cpu','gpu','finish']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='test':test()
    elif a.action=='smoke':smoke(a.device)
    elif a.action=='report':report()
    else:worker(a.lane,a.device)
