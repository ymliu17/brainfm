"""Common scans and subject splits across A424 and Schaefer400+Tian inputs."""
from pathlib import Path
import json,re,hashlib
import numpy as np,pandas as pd,torch
from model_input_protocol import build_input

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'runs/unified_input_v3/workspace'
DATA=ROOT/'data/derived/fm_posttraining_matrix_v2'
RUN=ROOT/'runs/fm_posttraining_matrix_v2'


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(obj),indent=2));t.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(name,rows,a,b):
    if not rows:raise ValueError('No matched rows '+name)
    from roi_signal_qc import assess
    for row,x,z in zip(rows,a,b):
        qa=assess(x[:,:min(int(row['native_frames']),x.shape[1])],200)
        qb=assess(z[:,:min(int(row['native_frames']),z.shape[1])],160)
        row.update(qc_dead_a424=qa['n_dead'],qc_dead_sch450=qb['n_dead'],qc_strict_pass=qa['strict_pass'] and qb['strict_pass'])
    tab=pd.DataFrame(rows);tab.to_csv(DATA/f'{name}.csv',index=False)
    torch.save(torch.stack(a),DATA/f'{name}_a424.pt');torch.save(torch.stack(b),DATA/f'{name}_sch450.pt')
    return dict(scans=len(rows),subjects=tab.subject.nunique(),a424_sha256=sha(DATA/f'{name}_a424.pt'),sch450_sha256=sha(DATA/f'{name}_sch450.pt'),manifest_sha256=sha(DATA/f'{name}.csv'))

def prepare():
    DATA.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'intake.json').exists():return
    counts={};exclusions=[]
    # ADNI manifests define clinical association/QC; intersect scans, not row positions.
    tables=[];tensors=[]
    for folder,prefix in [('adni424_t200','raw_ADNI_'),('adni450_t160','raw_450_ADNI_')]:
        blocks=[];ms=[]
        for target in ['MEM','EF']:
            tag=prefix+target+('_all_t200_off' if '424' in folder else '_all_t160_off');d=WORK/'data/derived'/folder
            m=pd.read_csv(d/f'ADNI_manifest_{tag}.csv');m['ix']=np.arange(len(m))+sum(len(z) for z in ms);ms.append(m);blocks.append(torch.load(d/f'ADNI_x_{tag}.pt',weights_only=True))
        tables.append(pd.concat(ms).drop_duplicates('FILE_NAME').set_index('FILE_NAME'));tensors.append(torch.cat(blocks))
    keys=sorted(set(tables[0].index)&set(tables[1].index));rows=[];aa=[];bb=[]
    for key in keys:
        a,b=[t.loc[key] for t in tables];x=tensors[0][int(a.ix)];z=tensors[1][int(b.ix)]
        if a.PTID!=b.PTID:raise ValueError('ADNI subject mismatch')
        for t in ['ADNI_MEM','ADNI_EF']:
            if pd.notna(a[t]) and pd.notna(b[t]) and not np.isclose(a[t],b[t]):raise ValueError('ADNI label mismatch')
        rows.append(dict(cohort='adni',subject=str(a.PTID),scan=key,visit=key.split('ses-')[-1],MEM=a.ADNI_MEM,EF=a.ADNI_EF,native_frames=min(a.n_timepoints_raw,b.n_timepoints_raw),eligible=True,age=a.AGE,sex=1. if a.PTGENDER=='Male' else 0. if a.PTGENDER=='Female' else np.nan))
        aa.append(x);bb.append(z)
    counts['adni']=write('adni',rows,aa,bb)
    # NACC and HRV: same selected clinical records, matched atlas QC.
    for name in ['nacc','hrv_pre','hrv_post']:
        d=WORK/'data/derived/external_preliminary_v1';m=pd.read_csv(d/f'{name}.csv');a=torch.load(d/f'{name}.pt',weights_only=True);rows=[];aa=[];bb=[]
        for i,r in m.iterrows():
            p=ROOT/str(r.source).replace('A424_noGSR','Schaefer400Tian_noGSR')
            try:
                z=build_input(torch.load(p,weights_only=True),160,qc_policy="sparse")
                sm=json.loads(p.with_suffix('.json').read_text())
                if sm.get('subject')!=r.subject:raise ValueError('Atlas subject mismatch')
                if str(sm.get('source_qc_exclude_recommend'))=='1':raise ValueError('Source QC')
            except (FileNotFoundError,ValueError) as e:exclusions.append(dict(cohort=name,subject=r.subject,reason=str(e)));continue
            row=dict(cohort='nacc' if name=='nacc' else 'hrv',subject=r.subject,scan=r.subject+'_'+str(r.session),visit=str(r.session),native_frames=r.native_frames,eligible=True)
            if name=='nacc':row.update(MEM=r.memory,EF=r.executive,age=r.age,sex=np.nan)
            else:
                for t in ['flanker','LSWM','PCPS']:
                    row[t]=r[('pre_' if name=='hrv_pre' else 'post_')+t];row['baseline_'+t]=r['pre_'+t];row['follow_'+t]=r['post_'+t]
                row.update(age_category=r.age_category,sex=r.sex,arm=r.arm)
            rows.append(row);aa.append(a[i]);bb.append(z)
        counts[name]=write(name,rows,aa,bb)
    # Use the already audited unified RCT loader, preserving clinical masks.
    import sys
    sys.path[:0]=[str(WORK/'src'),str(WORK/'scripts')]
    from cohorts import load_cohort
    from rct_positive_control import CONTROLS
    for name in ['act','cogte','breathe','psopt']:
        ca=load_cohort(name,release='amp',require_labels=False);cb=load_cohort(name,release='schaefer',require_labels=False);lookup={s:i for i,s in enumerate(cb.subjects)};ctrl=CONTROLS[name](ca.subjects);rows=[];aa=[];bb=[]
        for i,s in enumerate(ca.subjects):
            if s not in lookup:continue
            j=lookup[s]
            for tp in [0,1]:
                na,nb=int(ca.valid[i,tp]),int(cb.valid[j,tp])
                try:x=build_input(ca.bold[i,:,:na,tp],200,qc_policy="sparse");z=build_input(cb.bold[j,:,:nb,tp],160,qc_policy="sparse")
                except ValueError as e:exclusions.append(dict(cohort=name,subject=s,reason=str(e)));continue
                row=dict(cohort=name,subject=s,scan=s+'_T'+str(tp+1),visit='T'+str(tp+1),native_frames=min(na,nb),eligible=bool(ca.analysis_mask()[i]),age=ctrl['age'][0][i],sex=ctrl['sex'][0][i])
                for t in ['mem','ef']:
                    row[t.upper()]=(ca.baseline_levels(t) if tp==0 else ca.followup_levels(t))[i];row['baseline_'+t.upper()]=ca.baseline_levels(t)[i];row['follow_'+t.upper()]=ca.followup_levels(t)[i]
                rows.append(row);aa.append(x);bb.append(z)
        counts[name]=write(name,rows,aa,bb)
    # Add repeated NACC scans from the same eligible source population for SSL/identification.
    d=WORK/'data/derived/nacc_parity_v1';m=pd.read_csv(d/'repeated_scans.csv');a=torch.load(d/'repeated_scans.pt',weights_only=True);allowed=set(pd.read_csv(DATA/'nacc.csv').subject);rows=[];aa=[];bb=[]
    for i,r in m.iterrows():
        if r.PTID not in allowed:continue
        p=Path(str(r.source).replace('A424_noGSR','Schaefer400Tian_noGSR'))
        try:z=build_input(torch.load(p,weights_only=True),160,qc_policy="sparse")
        except (ValueError,FileNotFoundError):continue
        rows.append(dict(cohort='nacc',subject=r.PTID,scan=r.PTID+'_'+str(r.session),visit=str(r.session),native_frames=r.native_frames,eligible=True));aa.append(a[i]);bb.append(z)
    counts['nacc_repeated']=write('nacc_repeated',rows,aa,bb)
    save(RUN/'intake.json',dict(status='complete',counts=counts,exclusions=exclusions,scope='Common scan intersection across atlases; target-specific finite-label masks applied identically across all models. No implicit per-model inclusion changes.',upstream='Unified-input-v2 source files; spatial registration not independently verified'))


def dataset(name,model):
    suffix='sch450' if model=='brainjepa' else 'a424'
    manifest=DATA/f'{name}.csv';tensor=DATA/f'{name}_{suffix}.pt'
    expected=json.loads((RUN/'intake.json').read_text())['counts'][name]
    if sha(tensor)!=expected[suffix+'_sha256'] or sha(manifest)!=expected['manifest_sha256']:raise ValueError('Matrix input changed after registration: '+name)
    return torch.load(tensor,weights_only=True),pd.read_csv(manifest)


def endpoints(name):return ['flanker','LSWM','PCPS'] if name.startswith('hrv') else ['MEM','EF']


def splits(source,target,seed,mode):
    from sklearn.model_selection import GroupShuffleSplit
    from benchmark_evaluation import grouped_splits
    m=pd.read_csv(DATA/f'{source}.csv');ix=np.flatnonzero(m[target].notna());y=m[target].to_numpy()[ix];g=m.subject.to_numpy()[ix]
    pools=[(np.arange(len(ix)),np.array([],int))] if mode=='external' else list(grouped_splits(y,g,seed,5))
    out=[]
    for k,(pool,te) in enumerate(pools):
        a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=seed*100+k).split(pool,groups=g[pool]));tr,va=pool[a],pool[b]
        assert not set(g[tr])&set(g[va]) and not (set(g[tr])|set(g[va]))&set(g[te])
        out.append((ix[tr],ix[va],ix[te]))
    return out
