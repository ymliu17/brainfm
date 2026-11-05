"""Native-TR Schaefer17n400 inputs with explicit per-scan normalization variant."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from new_fm.data import source_rows, COHORTS, metadata
from fm_matrix.data import ROOT, WORK, save, sha, splits

RUN = ROOT/'runs/brainharmonix_f_v1'
DATA = ROOT/'data/derived/brainharmonix_f_v1'
MODELS = ('brainharmonix_f',)
STANDARD_SECONDS = 48 * .735
PATCHES = 18
# Packed tensors contain 400 signals and two metadata rows; metadata never enters encoder.
MAX_FRAMES = 2048

def transform(x, tr, model='brainharmonix_f', order=None):
    if x.ndim != 2 or x.shape[0] != 450 or not torch.isfinite(x).all():
        raise ValueError('Expected finite canonical Schaefer17n400+Tian50 signals')
    if not np.isfinite(tr) or tr <= 0: raise ValueError('Invalid native TR')
    patch = round(STANDARD_SECONDS / tr)
    if patch < 1 or PATCHES*patch > MAX_FRAMES: raise ValueError('Unsupported native TR')
    n = min(x.shape[1], PATCHES*patch)
    z = x[:400, :n].float()
    med = z.quantile(.5, dim=1, keepdim=True)
    iqr = z.quantile(.75, dim=1, keepdim=True)-z.quantile(.25, dim=1, keepdim=True)
    z = torch.where(iqr > 1e-6, (z-med)/iqr.clamp_min(1e-6), torch.zeros_like(z))
    out = torch.zeros(402, MAX_FRAMES)
    out[:400, :n] = z
    out[400, 0] = n
    out[401, 0] = patch
    out[401, 1] = tr
    return out, n

def prepare():
    DATA.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    if (RUN/'intake.json').exists():
        for v in json.loads((RUN/'intake.json').read_text())['counts'].values():
            for p,h in v['files_sha256'].items():
                if sha(p)!=h: raise ValueError('Changed registered input '+p)
        return
    parent=json.loads((ROOT/'runs/new_fm_v1/intake.json').read_text())
    counts={}; sources={}; checked=set()
    for co in COHORTS:
        p=DATA/f'{co}_brainharmonix_f.pt'; mp=DATA/(co+'.csv')
        m=metadata(co).copy(); ns=[]; patches=[]
        # Disk-backed allocation avoids holding all full-length cohorts in RAM.
        tmp=DATA/(co+'.build.npy')
        a=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=(len(m),402,MAX_FRAMES))
        for i,(x,tr,path) in enumerate(source_rows(co)):
            path=str(path)
            if path not in checked:
                h=sha(Path(path))
                if parent['sources_sha256'].get(path)!=h:raise ValueError('Unverified source '+path)
                sources[path]=h;checked.add(path)
            v,n=transform(x,tr); a[i]=v.numpy();ns.append(n);patches.append(int(v[401,0]))
        if len(ns)!=len(m):raise ValueError('Source row count mismatch')
        a.flush(); torch.save(torch.from_numpy(a),p);del a;tmp.unlink()
        m['input_real_frames']=ns;m['patch_size']=patches
        m['complete_context']=np.array(ns)==PATCHES*np.array(patches)
        # Compatibility column only: no resampling was performed.
        m['resampled_frames']=ns
        m.to_csv(mp,index=False)
        counts[co]=dict(scans=len(m),files_sha256={str(p):sha(p),str(mp):sha(mp)},
                       patch_sizes=sorted(set(patches)),complete_context_scans=int(m.complete_context.sum()))
        print('prepared',co,len(m),flush=True)
    save(DATA/'roi_mapping.json',dict(permutation=list(range(450)),
         verification='Official dataset uses Schaefer17n400; canonical project first400 ROIs are Schaefer17n400; no cortical reorder or Tian input.'))
    save(RUN/'intake.json',dict(status='complete',counts=counts,sources_sha256=parent['sources_sha256'],
        parent_intake_sha256=sha(ROOT/'runs/new_fm_v1/intake.json'),
        input='Schaefer17n400 native TR; patch size round(35.28/TR); up to18 patches; masked padding.',
        normalization='Disclosed benchmark variant: per-scan per-ROI median/IQR on retained real frames. Pretraining corpus statistics not supplied. No target-cohort fitted statistics.',
        evidence_status='benchmark_configuration; normalization is not an exact reproduction of pretraining',
        source_split_parent_sha256=sha(ROOT/'runs/fm_posttraining_matrix_v2/intake.json')))

def dataset(co,model):
    if model not in MODELS:raise ValueError(model)
    p=DATA/f'{co}_{model}.pt';mp=DATA/(co+'.csv')
    stamp=json.loads((RUN/'intake.json').read_text())['counts'][co]['files_sha256']
    for f in (p,mp):
        if sha(f)!=stamp[str(f)]:raise ValueError('Changed model input '+str(f))
    m=pd.read_csv(mp)
    if m.scan.tolist()!=metadata(co).scan.tolist():raise ValueError('Canonical row order changed')
    return torch.load(p,weights_only=True,mmap=True),m
