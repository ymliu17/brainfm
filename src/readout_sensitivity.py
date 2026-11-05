"""Frozen CLS/mean and JEPA regional readouts; one declared final layer only."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from fm_matrix.data import ROOT, save, sha
from fm_readout_v4.data import DATA, dataset, metadata
from benchmark_evaluation import grouped_splits, predict_prepared, score
from paper_completion import project_block, feature_dir
from paper_paired_stats import paired, holm

RUN=ROOT/'runs/readout_sensitivity_v1'
MODELS=('brainlm111','brainlm650','brainjepa')
COHORTS=('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post','nacc_repeated')
PAIR_COHORTS=('adni','nacc_repeated','act','cogte','breathe','psopt','hrv')


def read(p):return json.loads(Path(p).read_text())
def seed_of(key):return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
def main_folder(model):return feature_dir(('none',model,'frozen','none',0))
def mean_folder(model):return ROOT/'runs/fm_matched_v3/features'/(model+'_frozen')
def alternatives(model):return ('roi_flat','roi_norm') if model=='brainjepa' else ('content_mean',)


def audit():
    from brainjepa import DEFAULT_CKPT
    files={};shapes={}
    def register(p):files[str(p)]=sha(p)
    register(DEFAULT_CKPT)
    for co in COHORTS:
        m=metadata(co)
        if m.scan.duplicated().any():raise ValueError('Duplicate scan identifier')
        register(DATA/(co+'_domains.csv' if co.startswith('hrv') else co+'.csv'))
        register(DATA/(co+'_sch450.pt'))
        for model in MODELS:
            folders=[main_folder(model)]+([] if model=='brainjepa' else [mean_folder(model)])
            for folder in folders:
                stamp=read(folder/'complete.json');s=stamp['signature']
                if s['model']!=model or s.get('checkpoint_sha256') is not None or s.get('adapter_sha256') is not None:raise ValueError('Expected frozen checkpoint')
                if folder==mean_folder(model):
                    if s['readout']!='final_content_mean_v1' or s['implementation_sha256']!=sha(ROOT/'src/fm_matched_v3/models.py'):raise ValueError('Mean readout provenance mismatch')
                    if s['intake_sha256']!=sha(ROOT/'runs/fm_matched_v3/intake.json'):raise ValueError('Mean intake mismatch')
                elif s['intake_sha256']!=sha(ROOT/'runs/fm_posttraining_matrix_v2/intake.json'):raise ValueError('Native readout intake mismatch')
                p=folder/(co+'.npy');x=np.load(p,mmap_mode='r')
                if x.ndim!=2 or len(x)!=len(m) or not np.isfinite(x).all():raise ValueError('Invalid feature array')
                register(p);register(folder/'complete.json');shapes[str(p)]=list(x.shape)
    save(RUN/'input_audit.json',dict(status='complete',files_sha256=files,feature_shapes=shapes,
        registration='Existing current native-readout and content-mean arrays on the unchanged common-atlas input intake. New JEPA regional arrays verify reconstruction of the cached native mean.',
        limitation='Spatial registration and exact participant-level pretraining overlap remain unresolved in parent provenance audits.'))


def verify(paths):
    known=read(RUN/'input_audit.json')['files_sha256']
    for p in paths:
        if known.get(str(p))!=sha(p):raise ValueError('Registered input changed: '+str(p))


def roi_readouts(tokens):
    if tokens.ndim!=4 or tokens.shape[1:]!=(450,10,768):raise ValueError('Expected batch x 450 ROIs x 10 temporal tokens x 768 channels')
    roi=tokens.float().mean(2)
    return roi,roi.mean(1),torch.linalg.vector_norm(roi,dim=-1)


def extract_roi(co,device='cuda'):
    from fm_readout_v4.models import Encoder
    from fm_matrix.train import amp
    from brainjepa import DEFAULT_CKPT
    verify([DATA/(co+'_sch450.pt'),DEFAULT_CKPT,main_folder('brainjepa')/(co+'.npy')])
    x,_=dataset(co,'brainjepa');m=metadata(co);expected=np.load(main_folder('brainjepa')/(co+'.npy'))
    torch.manual_seed(0);net=Encoder('brainjepa',device);net.requires_grad_(False);net.eval()
    out=RUN/'roi_features'/co;out.mkdir(parents=True,exist_ok=True)
    temp=out/'roi.tmp.npy';roi_array=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(x),450,768))
    norms=[];native=[];errors=[]
    # Batch one matches the native mean exports and avoids changing inference precision.
    with torch.inference_mode():
        for i in range(len(x)):
            with amp(device):tokens=net.tokens(x[i:i+1].to(device))
            roi,mean,norm=roi_readouts(tokens)
            # Reproduce the old reduction in its original dtype before float conversion.
            original=tokens.mean((1,2)).float().cpu().numpy()[0]
            np.testing.assert_allclose(original,expected[i],rtol=1e-4,atol=1e-4)
            if not torch.isfinite(roi).all():raise ValueError('Nonfinite ROI embeddings')
            roi_array[i]=roi.cpu().numpy()[0];norms.append(norm.cpu().numpy()[0]);native.append(mean.cpu().numpy()[0])
            errors.append(float(np.max(np.abs(original-mean.cpu().numpy()[0]))))
            if i%50==0:print('JEPA ROI features',co,i,len(x),flush=True)
    roi_array.flush();del roi_array;temp.replace(out/'roi.npy')
    np.save(out/'roi_norm.npy',np.asarray(norms));np.save(out/'mean_from_roi.npy',np.asarray(native))
    save(out/'complete.json',dict(status='complete',cohort=co,scan_ids=m.scan.tolist(),shape=[len(x),450,768],
        feature_sha256={n:sha(out/n) for n in ('roi.npy','roi_norm.npy','mean_from_roi.npy')},
        checkpoint_sha256=sha(DEFAULT_CKPT),native_mean_reproduction='Cached native mean matched on every scan at rtol=atol=1e-4 using the original reduction dtype.',
        roi_mean_float32_vs_native_max_abs=float(max(errors)),
        definitions={'roi':'Final-layer 450 x 10 x 768 tokens; average ten temporal tokens per ROI in float32.',
                     'roi_flat':'Flatten the 450 x 768 matrix in ROI-major order; 345600 features before training-fold PCA.',
                     'roi_norm':'L2 norm of each temporally averaged regional feature vector; 450 features.',
                     'reference':'Existing final global mean, 768 features. No new encoder training or regional ordering inferred.'}))


def features(model,co):
    p=main_folder(model)/(co+'.npy');verify([p]);out={'native':np.load(p,mmap_mode='r')}
    if model!='brainjepa':
        p=mean_folder(model)/(co+'.npy');verify([p]);out['content_mean']=np.load(p,mmap_mode='r')
    else:
        folder=RUN/'roi_features'/co;stamp=read(folder/'complete.json')
        if stamp['scan_ids']!=metadata(co).scan.tolist():raise ValueError('ROI scan order mismatch')
        for key,name in [('roi_flat','roi.npy'),('roi_norm','roi_norm.npy')]:
            p=folder/name
            if sha(p)!=stamp['feature_sha256'][name]:raise ValueError('ROI feature changed')
            x=np.load(p,mmap_mode='r');out[key]=x.reshape(len(x),-1)
    return out


def evaluate(model,co,device='cpu'):
    m=metadata(co);verify([DATA/(co+'.csv')]);xs=features(model,co);out=RUN/'cognition'/model;out.mkdir(parents=True,exist_ok=True)
    for endpoint in ('MEM','EF'):
        dest=out/(co+'_'+endpoint+'.json')
        if dest.exists():continue
        ix=np.flatnonzero(m.eligible.to_numpy(bool)&np.isfinite(m[endpoint].to_numpy(float)))
        y=m.iloc[ix][endpoint].to_numpy(float);g=m.iloc[ix].subject.astype(str).to_numpy();byreadout={k:[] for k in xs};folds=[]
        for seed in (0,1,2):
            prepared={k:[] for k in xs}
            for fold,(tr,te) in enumerate(grouped_splits(y,g,seed)):
                it,iv=next(GroupShuffleSplit(1,test_size=.2,random_state=seed).split(tr,groups=g[tr]));it,iv=tr[it],tr[iv]
                for name,x in xs.items():
                    outer=project_block(x[ix[tr]],x[ix[te]],100,device)
                    inner=project_block(x[ix[it]],x[ix[iv]],100,device)
                    prepared[name].append((tr,te,outer,it,iv,inner))
                folds.append(dict(seed=seed,fold=fold,train=tr,test=te,inner_train=it,validation=iv))
            for name in xs:byreadout[name].append(predict_prepared(prepared[name],y,binary=False))
            print('readout cognitive probes',model,co,endpoint,seed,flush=True)
        predictions={k:np.mean(v,axis=0) for k,v in byreadout.items()};rows=[]
        for candidate in alternatives(model):
            stats=paired(y,predictions['native'],predictions[candidate],g,seed=seed_of(model+co+endpoint+candidate))
            rows.append(dict(model=model,cohort=co,endpoint=endpoint,reference='native',candidate=candidate,family='cognition_'+endpoint,**stats))
        save(dest,dict(status='complete',rows=rows,scan_ids=m.iloc[ix].scan.tolist(),subjects=g,y=y,predictions=predictions,
            predictions_by_seed=byreadout,metrics={k:dict(rho=score(y,p,False),mae=float(np.mean(np.abs(y-p)))) for k,p in predictions.items()},
            split_audit=folds,feature_dimensions={k:x.shape[1] for k,x in xs.items()},
            protocol='Identical grouped outer five-fold and grouped 20% inner validation splits; seeds 0/1/2. Training-only scaling and exact PCA <=100, fixed ridge grid. Primary paired scores use seed-averaged OOF predictions. Readouts fixed, not selected on test performance.'))
    save(out/(co+'_complete.json'),dict(status='complete',model=model,cohort=co))


def paired_rows(co):
    if co=='hrv':
        a,b=metadata('hrv_pre'),metadata('hrv_post')
        if a.subject.duplicated().any() or b.subject.duplicated().any():raise ValueError('Ambiguous HRV visits')
        common=sorted(set(a.subject)&set(b.subject))
        ia=a.reset_index().set_index('subject').loc[common,'index'].to_numpy();ib=b.reset_index().set_index('subject').loc[common,'index'].to_numpy()
        return common,('hrv_pre',ia),('hrv_post',ib)
    m=metadata(co);pairs=[];subjects=[]
    # Preserve the current identification population and first-two-visits pairing.
    for subject,g in m.groupby('subject',sort=True):
        if len(g)<2:continue
        visit=g.visit.astype(str)
        if co=='adni':
            order=visit.str.extract(r'^M(\d+)$',expand=False).astype(float)
            if order.isna().any():raise ValueError('Unrecognized ADNI visit')
            chosen=g.assign(_order=order).sort_values(['_order','scan']).index[:2].tolist()
        else:chosen=g.assign(_order=visit).sort_values(['_order','scan']).index[:2].tolist()
        pairs.append(chosen);subjects.append(str(subject))
    ia,ib=np.asarray(pairs,dtype=int).T
    return subjects,(co,ia),(co,ib)


def similarity(a,b,device='cpu'):
    # Correlation across feature dimensions; no fitted cross-participant transform.
    if device=='cuda':
        aa=torch.as_tensor(np.asarray(a),dtype=torch.float32,device=device);bb=torch.as_tensor(np.asarray(b),dtype=torch.float32,device=device)
        aa=aa-aa.mean(1,keepdim=True);bb=bb-bb.mean(1,keepdim=True)
        aa=aa/aa.norm(dim=1,keepdim=True).clamp_min(1e-8);bb=bb/bb.norm(dim=1,keepdim=True).clamp_min(1e-8)
        return (aa@bb.T).cpu().numpy()
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True)
    return (a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8))@(b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-8)).T


def identification_scores(sim):
    if sim.ndim!=2 or sim.shape[0]!=sim.shape[1] or not np.isfinite(sim).all():raise ValueError('Invalid identification matrix')
    n=len(sim);diag=np.diag(sim)
    # Fractional credit for exact maximum ties, avoiding participant-order bias.
    amax=sim==sim.max(1,keepdims=True);bmax=sim==sim.max(0,keepdims=True)
    a=amax.diagonal()/amax.sum(1);b=bmax.diagonal()/bmax.sum(0)
    diff=diag-(sim.sum(1)+sim.sum(0)-2*diag)/(2*(n-1))
    return dict(top1=float(((a+b)/2).mean()),pre_post=float(a.mean()),post_pre=float(b.mean()),
                chance=1/n,n=n,I_diff=float(diff.mean()),per_subject_top1=(a+b)/2,per_subject_I_diff=diff)


def paired_identification(a,b,seed=0,n_boot=5000,n_perm=20000):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2:raise ValueError('Unaligned identification scores')
    delta=b-a;rng=np.random.default_rng(seed);draws=[];exceed=0
    for start in range(0,n_boot,128):
        weights=rng.multinomial(len(a),np.full(len(a),1/len(a)),size=min(128,n_boot-start));draws.extend(weights@delta/len(a))
    for start in range(0,n_perm,128):
        signs=np.where(rng.random((min(128,n_perm-start),len(a)))<.5,1.,-1.)
        exceed+=int((np.abs(signs@delta/len(a))>=abs(delta.mean())-1e-12).sum())
    return dict(reference_score=float(a.mean()),candidate_score=float(b.mean()),delta=float(delta.mean()),ci95=np.quantile(draws,[.025,.975]),
        permutation_p=(1+exceed)/(1+n_perm),n_subjects=len(a),bootstrap_requested=n_boot,permutations_requested=n_perm,
        uncertainty='Participant bootstrap of bidirectional query scores conditional on fixed gallery; gallery not resampled.',
        test='Paired participant score swaps conditional on fixed gallery, exploratory; not a label-shuffle null or generalization to a new gallery.')


def identification(model,device='cpu'):
    out=RUN/'identification'/model;out.mkdir(parents=True,exist_ok=True)
    for co in PAIR_COHORTS:
        dest=out/(co+'.json')
        if dest.exists():continue
        subjects,(ca,ia),(cb,ib)=paired_rows(co)
        fa=features(model,ca);fb=fa if ca==cb else features(model,cb);scores={}
        for name in fa:scores[name]=identification_scores(similarity(fa[name][ia],fb[name][ib],device))
        rows=[]
        for candidate in alternatives(model):
            for metric in ('top1','I_diff'):
                stats=paired_identification(scores['native']['per_subject_'+metric],scores[candidate]['per_subject_'+metric],seed_of(model+co+candidate+metric))
                rows.append(dict(model=model,cohort=co,reference='native',candidate=candidate,metric=metric,family='identification_'+metric,**stats))
        save(dest,dict(status='complete',rows=rows,scores=scores,subjects=subjects,scan_ids_a=metadata(ca).iloc[ia].scan.tolist(),scan_ids_b=metadata(cb).iloc[ib].scan.tolist(),
            input_dimensions={k:f.shape[1] for k,f in fa.items()},
            protocol='Same participants and first two available visits for all readouts; HRV pre/post intersection. Native-dimensional correlation similarity; no fitted PCA or test-label readout selection. Not a dimension-matched identification experiment.'))
        print('identification',model,co,len(subjects),flush=True)
    save(out/'complete.json',dict(status='complete',model=model))


def report():
    rows=[]
    for p in [*(RUN/'cognition').glob('*/*.json'),*(RUN/'identification').glob('*/*.json')]:
        for r in read(p).get('rows',[]):rows.append(dict(r,input_file=str(p)))
    families={}
    for r in rows:families.setdefault(r['family'],[]).append(r)
    for family,rr in families.items():
        for r,p in zip(rr,holm([r['permutation_p'] for r in rr])):r.update(holm_p=p,family_size=len(rr))
    if len(rows)!=72:raise ValueError('Incomplete readout report: expected 16 cognitive and 56 identification contrasts')
    pd.json_normalize(rows).to_csv(RUN/'paired_comparisons.csv',index=False)
    for prefix in ('cognition','identification'):
        pd.json_normalize([r for r in rows if r['family'].startswith(prefix)]).to_csv(RUN/(prefix+'_comparisons.csv'),index=False)
    save(RUN/'report.json',dict(status='complete',n_comparisons=len(rows),families={k:len(v) for k,v in families.items()},rows=rows,
        scope='Fixed final-layer readout sensitivities in frozen models. BrainLM111/650 CLS vs existing content-weighted token mean; JEPA global mean vs full regional embeddings and regional norms.',
        caveats=['Exploratory readout comparisons; main readouts unchanged','Cognition matches PCA dimension budgets; native-dimensional identification does not isolate dimensionality effects','No post-training or learned readout selection','Parent spatial alignment and pretraining overlap limitations remain'],automatic_shutdown=False))
