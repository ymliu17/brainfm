"""Additional common-population controls; parent artifacts are read-only."""
from pathlib import Path
import json
import numpy as np
from fm_matrix.data import ROOT, save, sha
from fm_readout_v4.evaluate import tag
from fm_readout_v4.data import metadata
import paper_completion as pc
import sample_alignment as sa
import sf100_benchmark as sf
import readout_sensitivity as rs
from benchmark_evaluation import score
from fm_readout_v4.supplement import source_projection

RUN=ROOT/'runs/unified_supplements_v1'
CONFIGS=(*pc.MODELS,'semantoks','fmrilm','fmrilm_mae','fmrilm_vq_domain')
NEW=CONFIGS[4:]
PAIR_COHORTS=rs.PAIR_COHORTS

def read(p):return json.loads(Path(p).read_text())
def states(model):
    yield ('none',model,'frozen','none',0)
    for source in ('adni','nacc'):
        for obj,target in [('supervised','MEM'),('supervised','EF'),('native_ssl','none')]:
            for seed in (42,43,44):yield source,model,obj,target,seed

def feature_folder(state):
    source,model,obj,target,seed=state
    if model in pc.MODELS:return pc.feature_dir(state)
    if model in ('fmrilm_mae','fmrilm_vq_domain'):
        variant='mae' if model=='fmrilm_mae' else 'vq-domain'
        return ROOT/'runs/fmrilm_variants_v1'/variant/'features'/tag(source,'fmrilm',obj,target,seed)
    return ROOT/'runs/new_fm_v1/features'/tag(*state)

def feature(state,co):
    folder=feature_folder(state);p=folder/(co+'.npy');stamp=read(folder/'complete.json')
    if stamp['status']!='complete':raise ValueError('Incomplete features')
    if state[1] in pc.MODELS:pc.verify_input_files([p])
    elif stamp['feature_sha256'][co]!=sha(p):raise ValueError('Feature hash mismatch')
    x=np.load(p,mmap_mode='r')
    if len(x)!=len(metadata(co)):raise ValueError('Feature/metadata row mismatch')
    return x

def recall(state):
    """Identical one-way HRV recall design to run_hrv_recall_external.py."""
    train=('act','cogte','breathe','psopt');arrays=[];targets=[];subjects=[];audit=[];hashes={}
    for co in train:
        m=sf.labels(co);ix=np.flatnonzero(sa.common_mask(m,co)&np.isfinite(m.MEM.to_numpy(float)))
        y=m.iloc[ix].MEM.to_numpy(float);arrays.append(feature(state,co)[ix]);targets.append((y-y.mean())/max(y.std(),1e-8))
        subjects.extend((co+':'+m.iloc[ix].subject.astype(str)).tolist())
        audit.append(dict(cohort=co,scan_ids=m.iloc[ix].scan.tolist(),target_mean=y.mean(),target_sd=y.std()))
        hashes[str(feature_folder(state)/(co+'.npy'))]=sha(feature_folder(state)/(co+'.npy'))
    co='hrv_pre';m=sf.labels(co);ix=np.flatnonzero(sa.common_mask(m,co)&np.isfinite(m.MEM_delayed_recall.to_numpy(float)))
    pred=source_projection(np.concatenate(arrays),feature(state,co)[ix],np.concatenate(targets));y=m.iloc[ix].MEM_delayed_recall.to_numpy(float)
    parent=read(ROOT/'runs/hrv_recall_external_v1/sf100_fc.json')
    assert subjects==parent['training_subjects'] and m.iloc[ix].scan.tolist()==parent['scan_ids']
    np.testing.assert_array_equal(y,parent['y'])
    hashes[str(feature_folder(state)/(co+'.npy'))]=sha(feature_folder(state)/(co+'.npy'))
    save(RUN/'recall'/tag(*state)/'result.json',dict(status='complete',state=state,cohort=co,task='MEM_delayed_recall',
        scan_ids=m.iloc[ix].scan.tolist(),subjects=m.iloc[ix].subject.tolist(),y=y,prediction=pred,score=score(y,pred,False),
        metric='rho',n=len(y),training_subjects=subjects,training_audit=audit,feature_sha256=hashes,
        reference_sha256=sha(ROOT/'runs/hrv_recall_external_v1/sf100_fc.json'),
        protocol='Four-RCT standardized MEM -> HRV single Week-5 delayed recall from baseline MRI; source-only PCA<=100/ridge10. No target outcomes used in fitting.'))

def absolute_null(sim,seed,n_perm=20000):
    """Random bijections; shared mapping in both identification directions."""
    sim=np.asarray(sim,float);n=len(sim);idx=np.arange(n)
    rmax=sim==sim.max(1,keepdims=True);cmax=sim==sim.max(0,keepdims=True)
    credit=.5*(rmax/rmax.sum(1,keepdims=True)+cmax/cmax.sum(0,keepdims=True))
    top=float(np.diag(credit).mean());diff=float((n*np.trace(sim)-sim.sum())/(n*(n-1)))
    rng=np.random.default_rng(seed);hits=np.zeros(2,int);means=np.zeros(2)
    for start in range(0,n_perm,128):
        perms=np.stack([rng.permutation(n) for _ in range(min(128,n_perm-start))])
        ntop=credit[idx[None,:],perms].mean(1)
        ndiff=(n*sim[idx[None,:],perms].sum(1)-sim.sum())/(n*(n-1))
        vals=np.column_stack((ntop,ndiff));hits+=(vals>=np.array([top,diff])-1e-12).sum(0);means+=vals.sum(0)
    return {k:dict(score=v,exceedances=int(hits[i]),permutations=n_perm,p=(1+int(hits[i]))/(n_perm+1),
        null_mean=float(means[i]/n_perm),alternative='greater',test='Random identity bijection between visits; same mapping in both directions')
        for i,(k,v) in enumerate([('top1',top),('I_diff',diff)])}

def projected_similarity(a,b,dim=128,seeds=(0,1,2)):
    """Data-independent Gaussian projection; never fit PCA on test identities."""
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-8)
    sims=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);aa=np.zeros((len(a),dim));bb=np.zeros((len(b),dim))
        for start in range(0,a.shape[1],4096):
            stop=min(a.shape[1],start+4096);p=rng.standard_normal((stop-start,dim),dtype=np.float32)/np.sqrt(dim)
            aa+=a[:,start:stop]@p;bb+=b[:,start:stop]@p
        sims.append(rs.similarity(aa,bb))
    return np.mean(sims,axis=0)

def readouts(model,co):
    if model=='sf100_fc':return {'native':sf.features(co,include_fm=False)['sf100_fc']}
    if model=='brainlm13':
        from paper_task_expansion import readout_features
        return readout_features(co)
    if model in pc.MODELS:return rs.features(model,co)
    state=('none',model,'frozen','none',0);p=feature_folder(state)/(co+'_alternatives.npz')
    return {'native':feature(state,co),**dict(np.load(p))}

def identity(model,co):
    subjects,(ca,ia),(cb,ib)=sa.paired_rows(co);aa=readouts(model,ca);bb=aa if ca==cb else readouts(model,cb)
    fc_a=sf.features(ca,include_fm=False)['sf100_fc'][ia];fc_b=sf.features(cb,include_fm=False)['sf100_fc'][ib]
    refs={'native':rs.similarity(fc_a,fc_b),'projected128':projected_similarity(fc_a,fc_b)}
    rows=[];comparisons=[]
    for name in aa:
        a,b=aa[name][ia],bb[name][ib]
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite selected identity features')
        for mode,sim in [('native',rs.similarity(a,b)),('projected128',projected_similarity(a,b))]:
            scores=rs.identification_scores(sim);null=absolute_null(sim,rs.seed_of('identity-null:'+co))
            path=RUN/'identity'/model/co/(name+'_'+mode+'.npy');path.parent.mkdir(parents=True,exist_ok=True);np.save(path,sim)
            rows.append(dict(readout=name,mode=mode,dimensions=a.shape[1] if mode=='native' else 128,scores=scores,null=null,similarity_sha256=sha(path)))
            for metric in ('top1','I_diff'):
                ref=rs.identification_scores(refs[mode])['per_subject_'+metric]
                stat=rs.paired_identification(ref,scores['per_subject_'+metric],rs.seed_of(model+co+name+mode+metric))
                comparisons.append(dict(readout=name,mode=mode,metric=metric,reference='sf100_fc',**stat))
            print('Identity',model,co,name,mode,scores['top1'],flush=True)
    # Include the reference null once, without treating repeated FC tests as new evidence.
    references={mode:dict(scores=rs.identification_scores(sim),null=absolute_null(sim,rs.seed_of('identity-null:'+co))) for mode,sim in refs.items()}
    native={r['mode']:r for r in rows if r['readout']=='native'}
    readout_contrasts=[]
    for r in rows:
        if r['readout']=='native':continue
        for metric in ('top1','I_diff'):
            stat=rs.paired_identification(native[r['mode']]['scores']['per_subject_'+metric],r['scores']['per_subject_'+metric],rs.seed_of('readout'+model+co+r['readout']+r['mode']+metric))
            readout_contrasts.append(dict(readout=r['readout'],mode=r['mode'],metric=metric,reference='same_model_native_readout',**stat))
    save(RUN/'identity'/model/co/'result.json',dict(status='complete',model=model,cohort=co,subjects=subjects,
        scans_a=metadata(ca).iloc[ia].scan.tolist(),scans_b=metadata(cb).iloc[ib].scan.tolist(),rows=rows,
        comparisons=comparisons,readout_contrasts=readout_contrasts,references=references,
        inference='Raw tests only; publication requires joint all-configuration Holm families. Identity validity is distinct from cognitive utility.',
        protocol_sha256=sha(ROOT/'runs/unified_extension_v1/protocol.json')))
