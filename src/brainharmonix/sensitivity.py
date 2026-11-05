"""Fixed readout/context controls and dimensionality-matched FC fusion."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from brainharmonix.data import ROOT,RUN,DATA,source_rows,transform,metadata,save,sha
from brainharmonix.extract import folder,load
from brainharmonix.models import Encoder
from fm_matrix.train import amp
from paper_completion import project_block
from benchmark_evaluation import predict_prepared
import sample_alignment as sa

def short_features(model,co,device='cuda'):
    dest=RUN/'sensitivity/short_features'/model/(co+'.npy');stamp=dest.with_suffix('.json')
    if stamp.exists():return
    order=np.asarray(json.loads((DATA/'roi_mapping.json').read_text())['permutation'])
    m=metadata(co);ix=np.flatnonzero(sa.common_mask(m,co)&(m.native_frames>=80));needed=set(ix);xs=[]
    for i,(x,tr,_) in enumerate(source_rows(co)):
        if i in needed:xs.append(transform(x[:,:80],tr,model,order)[0])
    net=Encoder(model,device);values=[]
    with torch.no_grad():
        for start in range(0,len(xs),4):
            with amp(device):z=net(torch.stack(xs[start:start+4]).to(device))
            if not torch.isfinite(z).all():raise ValueError('Nonfinite short features')
            values.append(z.float().cpu().numpy())
    dest.parent.mkdir(parents=True,exist_ok=True);np.save(dest,np.concatenate(values))
    save(stamp,dict(status='complete',indices=ix,scan_ids=m.iloc[ix].scan.tolist(),feature_sha256=sha(dest),
        observed_native_frames=80,model_input_tr='native',scope='Same80 native frames; native TR patching; per-scan robust normalization variant.'))

def fusion(model,co,device='cpu'):
    import paper_completion as pc
    from paper_paired_stats import paired
    state=('none',model,'frozen','none',0);fm=load(state,co);out=RUN/'fusion'/model
    parent=(sa.PAPER if co=='cogte' else ROOT/'runs/paper_completion_v1')/'projections/sch450'/co
    for endpoint in pc.endpoints(co):
        dest=out/(co+'_'+endpoint+'.json')
        if dest.exists():continue
        meta=json.loads((parent/(endpoint+'.json')).read_text());fc=np.load(parent/(endpoint+'.npz'))
        with patch.object(pc,'primary_mask',sa.common_mask):m,ix,y,g,c=pc.fusion_sample(co,endpoint)
        if meta['indices']!=ix.tolist() or meta['subjects']!=g.tolist() or not np.array_equal(meta['y'],y):raise ValueError('Fusion population mismatch')
        x=fm[ix];budget=meta['budget']
        recipes={'FC':(budget,0),'FM':(0,budget),'FC+FM':(budget//2,budget//2)} if c is None else {'clinical':(0,0),'clinical+FC':(10,0),'clinical+FM':(0,10),'clinical+FC+FM':(5,5),'clinical+FC5':(5,0)}
        prepared={k:{s:[] for s in (0,1,2)} for k in recipes}
        for split in meta['splits']:
            tr,te,it,iv=[np.asarray(split[k]) for k in ('tr','te','it','iv')];key=split['key'];seed=split['seed']
            outer=project_block(x[tr],x[te],budget);inner=project_block(x[it],x[iv],budget);cs=[]
            for a,b in [(tr,te),(it,iv)]:
                if c is None:cs.append((np.empty((len(a),0)),np.empty((len(b),0))))
                else:
                    sc=StandardScaler().fit(c[a]);cs.append((sc.transform(c[a]),sc.transform(c[b])))
            for name,(kf,km) in recipes.items():
                pairs=[tuple(np.c_[cs[i][j],fc[key+label+('a' if j==0 else 'b')][:,:kf],ff[j][:,:km]] for j in (0,1)) for i,(label,ff) in enumerate([('outer',outer),('inner',inner)])]
                prepared[name][seed].append((tr,te,pairs[0],it,iv,pairs[1]))
        rows=[]
        for name,byseed in prepared.items():
            pred=[predict_prepared(byseed[s],y) for s in (0,1,2)]
            rows.append(pc.row(co,endpoint+('_baseline' if c is None else '_followup'),name,m,ix,y,np.mean(pred,axis=0),predictions_by_seed=pred,dimensions=recipes[name]))
        save(dest,dict(status='complete',rows=rows,split_audit=meta['splits'],
            reference='Existing Schaefer450 FC input pipeline; time context/resampling differ from new FM inputs, not claimed duration-matched.',
            protocol='Same original100 observational/10 follow-up total imaging dimensions, folds, participants, clinical covariates and tuning.'))
    save(out/(co+'_complete.json'),dict(status='complete'))

def robustness(model,co,device='cpu'):
    import paper_robustness as pr
    root=RUN/'sensitivity';parent=sa.RUN/'paper_robustness_v1'
    links={root/'labels':parent/'labels',root/'projections/sch450'/co:
        (parent if co=='cogte' else ROOT/'runs/paper_robustness_v1')/'projections/sch450'/co}
    for p,target in links.items():
        p.parent.mkdir(parents=True,exist_ok=True)
        try:p.symlink_to(target,target_is_directory=True)
        except FileExistsError:
            if p.resolve()!=target.resolve():raise ValueError('Unexpected sensitivity source')
    paths=[folder(('none',model,'frozen','none',0))/(co+'.npy'),root/'labels'/(co+'.csv')]
    checks={str(p):sha(p) for p in paths}
    def verify(ps):
        for p in ps:
            if str(p) not in checks or sha(p)!=checks[str(p)]:raise ValueError('Unregistered sensitivity input')
    with patch.multiple(pr,RUN=root,primary_mask=sa.common_mask,atlas_for=lambda _: 'sch450',feature_dir=folder,verify=verify):
        pr.evaluate(model,co)
    # The inherited >=200-native-frame restriction is reported as such, not mislabeled
    # as proof of no padding for new TR2 configurations.
    save(root/'evaluation'/model/co/'input_context_note.json',dict(status='complete',
        inherited_no_padding_subset='Original benchmark native>=200 subset; not equivalent to complete18patch context.',
        actual_complete_context_scans=int(pd.read_csv(DATA/(co+'.csv')).complete_context.sum())))
