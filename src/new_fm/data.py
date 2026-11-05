"""Original-length Schaefer450 signals, canonical rows, explicit TR2 input variants."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from fm_matrix.data import ROOT, WORK, save, sha, splits
from fm_readout_v4.data import metadata
from temporal import resample_time

RUN = ROOT/'runs/new_fm_v1'
DATA = ROOT/'data/derived/new_fm_v1'
COHORTS = ('adni','nacc','act','cogte','breathe','psopt','hrv_pre','hrv_post','nacc_repeated')
MODELS = ('semantoks','fmrilm')

def mapping():
    """Verify the 17→7 cortical permutation by exact parcel voxel identity."""
    import nibabel as nib
    folder=DATA/'atlases'
    paths=[folder/f'Schaefer2018_400Parcels_{n}Networks_order_FSLMNI152_2mm.nii.gz' for n in (17,7)]
    a,b=[nib.load(str(p)) for p in paths]
    if a.shape!=b.shape or not np.allclose(a.affine,b.affine): raise ValueError('Atlas grid mismatch')
    x,y=[np.asarray(i.dataobj).astype(int) for i in (a,b)]
    order=[]
    for k in range(1,401):
        labels=np.unique(x[y==k])
        if len(labels)!=1 or labels[0]==0 or not np.array_equal(x==labels[0],y==k):
            raise ValueError('Not an exact Schaefer permutation')
        order.append(int(labels[0])-1)
    if len(set(order))!=400:raise ValueError('Not bijective')
    # Match the project's declared 17-network centroid ordering to the downloaded atlas.
    coord=np.loadtxt(ROOT/'src/toolkit/atlases/Schaefer400Tian_S3_Coordinates.dat')[:400,1:4]
    cp=folder/'Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
    cent=pd.read_csv(cp)[['R','A','S']].to_numpy()
    if not np.array_equal(coord,cent):raise ValueError('Project cortical ROI order differs from official centroid table')
    paths.append(cp)
    save(DATA/'roi_mapping.json',dict(status='complete',permutation=order+list(range(400,450)),
        source='Schaefer400 17Networks + Tian-S3 label order',target='Schaefer400 7Networks + Tian-S3 label order',
        verification='Exact cortical parcel voxel identity and project centroids; upstream extraction order inherited from declared project atlas.',
        files_sha256={str(p):sha(p) for p in paths}))
    return np.array(order+list(range(400,450)))

def source_rows(co):
    m=metadata(co)
    if co in ('act','cogte','breathe','psopt'):
        from cohorts import load_cohort
        c=load_cohort(co,release='schaefer',require_labels=False)
        lookup={str(s):i for i,s in enumerate(c.subjects)}
        for r in m.itertuples():
            i=lookup[str(r.subject)];v={'T1':0,'T2':1}[r.visit];n=int(c.valid[i,v])
            yield c.bold[i,:,:n,v],float(c.native_tr),str(c.bold_path)
    elif co=='adni':
        tables=pd.concat([pd.read_csv(WORK/f'data/derived/adni450_t160/ADNI_manifest_raw_450_ADNI_{t}_all_t160_off.csv') for t in ('MEM','EF')]).drop_duplicates('FILE_NAME').set_index('FILE_NAME')
        for r in m.itertuples():
            paths=list((ROOT/'data/adni/raw_450').glob(r.scan+'_*.pt'))
            if len(paths)!=1:raise ValueError('Ambiguous ADNI source '+r.scan)
            yield torch.load(paths[0],weights_only=True),float(tables.loc[r.scan,'native_tr']),str(paths[0])
    else:
        if co=='nacc_repeated':
            t=pd.read_csv(WORK/'data/derived/nacc_parity_v1/repeated_scans.csv')
            t['scan']=t.PTID+'_'+t.session.astype(str)
        else:
            t=pd.read_csv(WORK/f'data/derived/external_preliminary_v1/{co}.csv')
            t['scan']=t.subject+'_'+t.session.astype(str)
        t=t.set_index('scan')
        for r in m.itertuples():
            p=str(t.loc[r.scan,'source']).replace(str(WORK),str(ROOT)).replace('A424_noGSR','Schaefer400Tian_noGSR')
            p=Path(p) if Path(p).is_absolute() else ROOT/p
            side=json.loads(p.with_suffix('.json').read_text())
            yield torch.load(p,weights_only=True),float(side['tr']),str(p)

def transform(x,tr,model,order):
    if x.ndim!=2 or x.shape[0]!=450 or not torch.isfinite(x).all():raise ValueError('Invalid native timeseries')
    if not np.isfinite(tr) or tr<=0:raise ValueError('Missing native TR')
    x=resample_time(x.float(),tr,2.)
    # Keep up to 180 frames for self-distillation crops; scan-level inference uses first100/160.
    # Official fMRI-LM TianS3 combined atlas labels1..50 are subcortex,
    # labels51..450 are Schaefer7-network cortex. Semantoks HDF5 concatenates
    # cortex, subcortex, cerebellum instead. Equal ROI counts are insufficient.
    input_order=order if model=='semantoks' else np.r_[order[400:],order[:400]]
    n=min(x.shape[1],180);x=x[input_order,:n]
    if model=='semantoks':
        x=(x-x.mean(1,keepdim=True))/x.std(1,unbiased=False,keepdim=True).clamp_min(1e-8)
        x=torch.cat((x,torch.zeros(7,n)))
    else:
        # Released cohort normalization_params.npz is absent. Label-free, per-scan
        # per-ROI robust scaling is an explicit benchmark variant, not an exact reproduction.
        med=x.quantile(.5,dim=1,keepdim=True);iqr=x.quantile(.75,dim=1,keepdim=True)-x.quantile(.25,dim=1,keepdim=True)
        x=(x-med)/iqr.clamp_min(1e-6)
        x=torch.where(iqr>1e-6,x,torch.zeros_like(x))
    return torch.nn.functional.pad(x,(0,180-n)),n

def prepare():
    if (RUN/'intake.json').exists():
        for count in json.loads((RUN/'intake.json').read_text())['counts'].values():
            for p,h in count['files_sha256'].items():
                if sha(p)!=h:raise ValueError('Registered model input changed '+p)
        return
    DATA.mkdir(parents=True,exist_ok=True);order=mapping();counts={};sources={}
    for co in COHORTS:
        paths=[DATA/f'{co}_{model}.pt' for model in MODELS];mp=DATA/(co+'.csv')
        if all(p.exists() for p in paths) and mp.exists():
            counts[co]=dict(scans=len(pd.read_csv(mp)),files_sha256={str(p):sha(p) for p in paths+[mp]});continue
        m=metadata(co).copy();out={name:[] for name in MODELS};frames=[];trs=[];native=[]
        for i,(x,tr,path) in enumerate(source_rows(co)):
            if x.shape[1]<int(m.iloc[i].native_frames):raise ValueError('Native source shorter than registered scan')
            native.append(x.shape[1]);trs.append(tr)
            for model in MODELS:
                y,n=transform(x,tr,model,order);out[model].append(y)
            frames.append(n)
            if path not in sources:sources[path]=sha(path)
        if len(frames)!=len(m):raise ValueError('Canonical scan count changed')
        m['resampled_frames']=frames;m['native_tr']=trs;m['source_frames']=native;m.to_csv(mp,index=False)
        for model,path in zip(MODELS,paths):
            tmp=path.with_suffix('.tmp.pt');torch.save(torch.stack(out[model]),tmp);tmp.replace(path)
        counts[co]=dict(scans=len(m),files_sha256={str(p):sha(p) for p in paths+[mp]})
        print('prepared',co,len(m),'resampled frames',min(frames),max(frames),flush=True)
    save(RUN/'intake.json',dict(status='complete',counts=counts,sources_sha256=sources,
        population='Canonical matrix rows, unchanged labels; SF100 task masks applied in evaluation; no new QC exclusions.',
        input='Polyphase TR2 resampling of original-length signals. Up to180 frames stored; first100 Semantoks / first160 fMRI-LM at inference; right zero padding after scan-local normalization.',
        semantoks='Teacher CLS; exact 17→7 cortical permutation; missing Buckner7 tokens replaced with pretrained mask embedding.',
        fmrilm='Stage2 vq-contrastive brain-only Qwen mean. Per-scan per-ROI median/IQR variant because released cohort normalization statistics are absent; no cohort/test-fitted normalization.',
        evidence_status='benchmark_configuration; not exact fMRI-LM preprocessing reproduction',
        source_split_parent_sha256=sha(ROOT/'runs/fm_posttraining_matrix_v2/intake.json')))

def dataset(co,model):
    if model not in MODELS:raise ValueError(model)
    p=DATA/f'{co}_{model}.pt';m=pd.read_csv(DATA/(co+'.csv'))
    stamp=json.loads((RUN/'intake.json').read_text())['counts'][co]['files_sha256']
    for f in (p,DATA/(co+'.csv')):
        if sha(f)!=stamp[str(f)]:raise ValueError('Input changed '+str(f))
    if m.scan.tolist()!=metadata(co).scan.tolist():raise ValueError('Canonical row order changed')
    return torch.load(p,weights_only=True),m
