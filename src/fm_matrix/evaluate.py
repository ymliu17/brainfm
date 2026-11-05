"""Matched frozen/adapted external axes, local probes, prognosis and physiology."""
from pathlib import Path
import json,re,hashlib
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from fm_matrix.data import ROOT,WORK,DATA,RUN,dataset,endpoints,save,sha
from fm_matrix.models import Encoder
from fm_matrix.train import features,location,amp
from benchmark_evaluation import oof_predictions,paired_bootstrap,score
from transfer_baseline import fc_feature,amp_feature
from model_input_protocol import build_input

COHORTS=['adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post']


def checkpoint_path(source,model,objective,target,seed):
    p=location(source,model,objective,target,seed,'external')
    return p/'fold0/adapter.pt' if objective=='supervised' else p/'adapter.pt'


def tag(source,model,objective,target,seed):return model+'_'+('frozen' if objective=='frozen' else f'{source}_{objective}_{target}_s{seed}')


def feature_dir(source,model,objective,target,seed):return RUN/'features'/tag(source,model,objective,target,seed)


def extraction(source,model,objective,target,seed,device='cuda'):
    out=feature_dir(source,model,objective,target,seed);ck=None if objective=='frozen' else checkpoint_path(source,model,objective,target,seed)
    signature=dict(intake_sha256=sha(RUN/'intake.json'),adapter_sha256=sha(ck) if ck else None,model=model)
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['signature']!=signature:raise ValueError('Stale matrix feature cache')
        return
    net=Encoder(model,device)
    if ck:
        c=torch.load(ck,weights_only=True)
        if c['model']!=model or c['intake_sha256']!=signature['intake_sha256'] or c['source_checkpoint_sha256']!=sha(net.checkpoint_path):raise ValueError('Checkpoint provenance mismatch')
        net.load_adapter(c['adapter'])
    out.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for name in COHORTS+['nacc_repeated']:
            p=out/(name+'.npy')
            if not p.exists():x,m=dataset(name,model);np.save(p,features(net,x,device))
        for name in ['breathe','psopt','hrv_pre','hrv_post']:
            p=out/(name+'_windows.npy')
            if p.exists():continue
            x,m=dataset(name,model);keep=np.flatnonzero(m.native_frames>=160);xx=torch.stack([build_input(x[i,:,:160],net.length) for i in keep]);vals=[]
            for i in range(len(xx)):
                with amp(device):v=net.windows(xx[i:i+1].to(device))
                if not torch.isfinite(v).all():raise ValueError('Nonfinite window features')
                vals.append(v.float().cpu().numpy())
            np.save(p,np.concatenate(vals));np.save(out/(name+'_window_indices.npy'),keep)
    save(out/'complete.json',dict(status='complete',signature=signature,encoder=net.provenance(),physiology='Eight full 20-TR windows over first 160 real frames; overlap-weighted native temporal tokens; model-native context length retained by padding only; offline contextual features, not causal online prediction'))


def baselines(model):
    atlas='sch450' if model=='brainjepa' else 'a424';d=RUN/'baselines'/atlas;d.mkdir(parents=True,exist_ok=True)
    for name in COHORTS+['nacc_repeated']:
        if (d/(name+'_FC.npy')).exists():continue
        x,_=dataset(name,model);np.save(d/(name+'_FC.npy'),fc_feature(x.numpy(),None).astype('float32'));np.save(d/(name+'_amp.npy'),amp_feature(x.numpy(),None).astype('float32'))
    return d


def array(folder,name):return np.load(folder/(name+'.npy'),mmap_mode='r')

def oof(F,y,g,k=100):
    # Cache exact, matched frozen/FC predictions rather than refit them for every state.
    h=hashlib.sha256();h.update(np.ascontiguousarray(F).tobytes());h.update(np.asarray(y,dtype=np.float64).tobytes());h.update(json.dumps([str(v) for v in g]).encode());h.update(str(k).encode())
    p=RUN/'prediction_cache'/(h.hexdigest()+'.npy');p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():return np.load(p)
    v=np.mean([oof_predictions([F],[k],y,s,groups=g) for s in [0,1,2]],axis=0);tmp=p.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(p);return v


def fitted_axis(train,y,test):
    sc=StandardScaler().fit(train);a=sc.transform(train);pc=PCA(n_components=min(100,len(train)-1,train.shape[1]),svd_solver='randomized',random_state=0).fit(a)
    return Ridge(alpha=10).fit(pc.transform(a),y).predict(pc.transform(sc.transform(test)))

def comparison(y,a,b,g):
    delta,p,ci,detail=paired_bootstrap(y,a,b,n_boot=2000,groups=g,return_details=True)
    return dict(delta_rho=delta,ci95=ci,raw_one_sided_bootstrap_p=p,**detail)

def row(name,task,y,p,g,**kw):return dict(cohort=name,task=task,n=len(y),subjects=g,y=y,prediction=p,rho=score(y,p,False),**kw)


def clinical(m,name,target):
    if name.startswith('hrv'):
        cat=pd.get_dummies(m[['age_category','sex','arm']],drop_first=True,dtype=float).to_numpy()
        return np.column_stack([m['baseline_'+target],cat])
    return m[['baseline_MEM','baseline_EF','age','sex']].to_numpy(float)


def cognitive(source,model,objective,target,seed,qc_strict=False):
    f=feature_dir(source,model,objective,target,seed);frozen=feature_dir('none',model,'frozen','none',0);b=baselines(model);results=[];comparisons=[]
    out=RUN/'evaluation'/tag(source,model,objective,target,seed)/('cognitive_strict.json' if qc_strict else 'cognitive.json')
    if out.exists():return
    for name in COHORTS:
        if name=='hrv_post' or (objective!='frozen' and name==source):continue
        m=pd.read_csv(DATA/f'{name}.csv');F=array(f,name);F0=array(frozen,name);FC=array(b,name+'_FC');eligible=m.eligible.to_numpy(bool)
        if qc_strict:eligible &= m.qc_strict_pass.to_numpy(bool)
        if name in ['act','cogte','breathe','psopt']:eligible &= m.visit.eq('T1').to_numpy()
        for task in endpoints(name):
            y=m[task].to_numpy(float);ix=np.flatnonzero(eligible&np.isfinite(y));g=m.subject.to_numpy()[ix]
            if len(np.unique(g))<20:continue
            yy=y[ix];preds={}
            for method,z in [('FM',F),('FC',FC)]:preds[method]=oof(z[ix],yy,g)
            preds['FC+FM']=np.mean([oof_predictions([FC[ix],F[ix]],[50,50],yy,s,groups=g) for s in [0,1,2]],axis=0)
            for method,p in preds.items():results.append(row(name,task+'_baseline',yy,p,g,method=method))
            for a,bb in [('FC','FM'),('FC','FC+FM')]:comparisons.append(dict(cohort=name,task=task+'_baseline',reference=a,test=bb,**comparison(yy,preds[a],preds[bb],g)))
            if objective!='frozen':
                p0=oof(F0[ix],yy,g);comparisons.append(dict(cohort=name,task=task+'_baseline',reference='frozen',test='adapted',**comparison(yy,p0,preds['FM'],g)))
            if 'follow_'+task in m:
                C=clinical(m,name,task);follow=m['follow_'+task].to_numpy(float);ii=np.flatnonzero(eligible&np.isfinite(follow)&np.isfinite(C).all(1));gg=m.subject.to_numpy()[ii];yy=follow[ii]
                if len(np.unique(gg))>=20:
                    pp={}
                    for method,blocks,dims in [('clinical',[C],[C.shape[1]]),('clinical+FC',[C,FC],[C.shape[1],5]),('clinical+FM',[C,F],[C.shape[1],5]),('clinical+FC+FM',[C,FC,F],[C.shape[1],2,3])]:
                        pp[method]=np.mean([oof_predictions([z[ii] for z in blocks],dims,yy,s,groups=gg) for s in [0,1,2]],axis=0);results.append(row(name,task+'_followup',yy,pp[method],gg,method=method))
                    for a,bb in [('clinical','clinical+FM'),('clinical+FC','clinical+FC+FM')]:comparisons.append(dict(cohort=name,task=task+'_followup',reference=a,test=bb,**comparison(yy,pp[a],pp[bb],gg)))
                    if objective!='frozen':
                        p0=np.mean([oof_predictions([C[ii],F0[ii]],[C.shape[1],5],yy,s,groups=gg) for s in [0,1,2]],axis=0);comparisons.append(dict(cohort=name,task=task+'_followup',reference='frozen',test='adapted',**comparison(yy,p0,pp['clinical+FM'],gg)))
    # Fixed source axes on independent target cohorts, separate from local probes.
    sources=['adni','nacc'] if objective=='frozen' else [source]
    for src in sources:
        m=pd.read_csv(DATA/f'{src}.csv');sf=array(f,src)
        for domain in ['MEM','EF']:
            keep=m[domain].notna().to_numpy()
            if qc_strict:keep &= m.qc_strict_pass.to_numpy(bool)
            for name in COHORTS:
                if name==src or name=='hrv_post':continue
                tm=pd.read_csv(DATA/f'{name}.csv');sel=tm.eligible.to_numpy(bool)
                if qc_strict:sel &= tm.qc_strict_pass.to_numpy(bool)
                if name in ['act','cogte','breathe','psopt']:sel &= tm.visit.eq('T1').to_numpy()
                p=fitted_axis(sf[keep],m.loc[keep,domain].to_numpy(),array(f,name))
                for task in (endpoints(name) if name.startswith('hrv') else [domain]):
                    y=tm[task].to_numpy(float);ii=sel&np.isfinite(y);results.append(row(name,src+'_'+domain+'_axis_to_'+task,y[ii],p[ii],tm.loc[ii,'subject'].to_numpy(),method='source_axis'))
    save(out,dict(status='complete',model=model,source=source,objective=objective,rows=results,comparisons=comparisons,roi_qc_sensitivity='Zero-constant-ROI common-atlas subset; probes and source axes refitted; encoders trained on primary population' if qc_strict else 'Primary sparse-ROI policy',source_evaluation='Adapted encoders excluded from local evaluation in their source cohort',multiplicity='Exploratory; raw paired-bootstrap tails. Pooled correction in matrix report.'))


def identification(source,model,objective,target,seed):
    f=feature_dir(source,model,objective,target,seed);rows=[]
    for name in ['adni','nacc_repeated','act','cogte','breathe','psopt','hrv']:
        if objective!='frozen' and name.removesuffix('_repeated')==source:continue
        if name=='hrv':
            a=pd.read_csv(DATA/'hrv_pre.csv');b=pd.read_csv(DATA/'hrv_post.csv');common=sorted(set(a.subject)&set(b.subject));ia=a.reset_index().set_index('subject').loc[common,'index'];ib=b.reset_index().set_index('subject').loc[common,'index'];A=array(f,'hrv_pre')[ia];B=array(f,'hrv_post')[ib]
        else:
            m=pd.read_csv(DATA/f'{name}.csv');pairs=[g.sort_values('visit').index[:2].to_list() for _,g in m.groupby('subject') if len(g)>=2]
            if not pairs:continue
            ia,ib=np.asarray(pairs).T;z=array(f,name);A=z[ia];B=z[ib]
        A=A-A.mean(1,keepdims=True);B=B-B.mean(1,keepdims=True);A=A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-8);B=B/np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-8);sim=A@B.T
        rows.append(dict(cohort=name,n=len(A),chance=1/len(A),pre_post=float(np.mean(sim.argmax(1)==np.arange(len(A)))),post_pre=float(np.mean(sim.argmax(0)==np.arange(len(A))))))
    save(RUN/'evaluation'/tag(source,model,objective,target,seed)/'identification.json',dict(status='complete',rows=rows,scope='Cross-visit identification; intervention visits are not intervention-free retest reliability'))


def loso(source,model,objective,target,seed):
    f=feature_dir(source,model,objective,target,seed);rows=[];names=['act','cogte','breathe','psopt']
    for domain in ['MEM','EF']:
        for held in names:
            train=[];ys=[]
            for name in names:
                if name==held:continue
                m=pd.read_csv(DATA/f'{name}.csv');ok=m.eligible&m.visit.eq('T1')&m[domain].notna();y=m.loc[ok,domain].to_numpy();ys.append((y-y.mean())/max(y.std(),1e-8));train.append(array(f,name)[ok])
            m=pd.read_csv(DATA/f'{held}.csv');ok=m.eligible&m.visit.eq('T1')&m[domain].notna();y=m.loc[ok,domain].to_numpy();p=fitted_axis(np.concatenate(train),np.concatenate(ys),array(f,held)[ok]);obs=score(y,p,False);rng=np.random.default_rng(0);null=[score(rng.permutation(y),p,False) for _ in range(1000)]
            rows.append(row(held,domain,y,p,m.loc[ok,'subject'].to_numpy(),permutation_p=(1+sum(abs(v)>=abs(obs) for v in null))/1001))
    save(RUN/'evaluation'/tag(source,model,objective,target,seed)/'loso.json',dict(status='complete',rows=rows,scope='Source-fitted feature scaling/PCA; training-cohort outcome standardization; held-out cohort labels used only for final score/permutation'))


def physiology(source,model,objective,target,seed):
    f=feature_dir(source,model,objective,target,seed);f0=feature_dir('none',model,'frozen','none',0);rows=[];comparisons=[]
    from rct_inscanner_patch_hr import subj_key,COHORTS as HR
    from hrv_ppg_benchmark import valid_window_rows
    for name in ['breathe','psopt','hrv']:
        if name=='hrv':w=pd.read_csv(ROOT/'data/HRV_biofeedback/label/HRV_biofeedback_PPG_HRV_windows48s_full.csv');w=w.loc[valid_window_rows(w)&w.WindowIdx.between(0,7)].copy();w['key']=w.Subject;w['visit']=w.Timepoint;targets={'pulse_rate':'HRV_MeanHR','log_RMSSD':'HRV_RMSSD'};ds=['hrv_pre','hrv_post']
        else:
            w=pd.read_csv(ROOT/HR[name]['hr']);w=w[w.Usable_HR&w.MeanHR.between(40,120)&w.WindowIdx.between(0,7)].copy()
            if 'Run' in w:w=w[w.Run.eq('rest-run01')]
            w['key']=w.Subject.map(subj_key);w['visit']=w.Timepoint;targets={'heart_rate':'MeanHR'};ds=[name]
        lookup={};lookup0={};groups={}
        for d in ds:
            m=pd.read_csv(DATA/f'{d}.csv');idx=np.load(f/(d+'_window_indices.npy'));z=array(f,d+'_windows');z0=array(f0,d+'_windows')
            for j,i in enumerate(idx):
                r=m.iloc[i];key=r.subject if name=='hrv' else subj_key(r.subject);visit=r.visit;lookup[key,visit]=z[j];lookup0[key,visit]=z0[j];groups[key,visit]=r.subject
        for task,col in targets.items():
            X=[];X0=[];Y=[];G=[];scan=[]
            for (key,visit),tab in w.groupby(['key','visit']):
                if (key,visit) not in lookup or sorted(tab.WindowIdx.tolist())!=list(range(8)):continue
                y=tab.sort_values('WindowIdx')[col].to_numpy(float)
                if task.startswith('log_'):
                    if (y<=0).any():continue
                    y=np.log(y)
                if not np.isfinite(y).all():continue
                X.append(lookup[key,visit]);X0.append(lookup0[key,visit]);Y.extend(y-y.mean());G.extend([groups[key,visit]]*8);scan.extend([str(key)+'_'+visit]*8)
            if len(set(G))<5:
                rows.append(dict(cohort=name,task=task,status='insufficient_subjects',subjects=len(set(G))));continue
            X=np.concatenate(X);X0=np.concatenate(X0);y=np.asarray(Y);g=np.asarray(G)
            from benchmark_evaluation import prepare_nested,predict_prepared
            prepared=[prepare_nested([X],[10],np.zeros(len(y)),g,s) for s in [0,1,2]]
            def physiology_predict(yy):return np.mean([predict_prepared(v,yy,binary=False) for v in prepared],axis=0)
            p=physiology_predict(y);rr=row(name,task,y,p,g,scan=scan)
            # Circular-shift null refits the predictor and preserves temporal ordering within each scan.
            rng=np.random.default_rng(0);null=[]
            for _ in range(200):
                yy=y.copy()
                for s in np.unique(scan):
                    ix=np.flatnonzero(np.asarray(scan)==s);yy[ix]=np.roll(y[ix],int(rng.integers(1,len(ix))))
                null.append(score(yy,physiology_predict(yy),False))
            rr['circular_shift_p']=(1+sum(abs(v)>=abs(rr['rho']) for v in null))/201;rows.append(rr)
            if objective!='frozen':
                p0=np.mean([predict_prepared(prepare_nested([X0],[10],np.zeros(len(y)),g,s),y,binary=False) for s in [0,1,2]],axis=0)
                comparisons.append(dict(cohort=name,task=task,**comparison(y,p0,p,g)))
    save(RUN/'evaluation'/tag(source,model,objective,target,seed)/'physiology.json',dict(status='complete',rows=rows,comparisons=comparisons,scope='Within-scan contextual association, participant-grouped probes; HRV is PPG-derived PRV. Circular-shift null assumes within-scan stationarity; no causal interpretation.'))


def cognitive_strict(source,model,objective,target,seed):
    return cognitive(source,model,objective,target,seed,qc_strict=True)
