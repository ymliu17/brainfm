"""Task-specific common-scan refits, preserving registered parent results.

Only SF100 availability changes eligibility. Missing follow-up imaging affects
identification, not prediction from a baseline scan with a follow-up outcome.
Encoders and source training partitions are unchanged. No new endpoints.
"""
from pathlib import Path
from functools import lru_cache
import json
import shutil
import hashlib
from collections import Counter
import numpy as np
import pandas as pd
from fm_matrix.data import ROOT, save, sha
from fm_readout_v4.data import metadata, endpoints
import paper_completion as pc
import demographic_benchmark as demo
import sf100_benchmark as sf
from benchmark_evaluation import prepare_nested, predict_prepared, score
from fm_matched_v3.probes import oof, oof_predictions

RUN = ROOT/'runs/sample_alignment_v1'
PAPER = RUN/'paper_completion_v1'
PARENT_PAPER = pc.RUN
PARENT_DEMO = demo.RUN
MODELS = pc.MODELS


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(None)
def available(co):
    rows = sf.matches(co)
    m = metadata(co)
    if [r['scan'] for r in rows] != m.scan.astype(str).tolist():
        raise ValueError('SF100/canonical scan order mismatch: '+co)
    return np.asarray([r['available'] for r in rows], bool)


def common_mask(m, co):
    if m.scan.astype(str).tolist() != metadata(co).scan.astype(str).tolist():
        raise ValueError('Canonical row order required')
    keep = m.eligible.to_numpy(bool).copy() & available(co)
    if co in ('act', 'cogte', 'breathe', 'psopt'):
        keep &= m.visit.eq('T1').to_numpy()
    return keep


def common_metadata(co):
    m = metadata(co).copy()
    m['eligible'] = m.eligible.to_numpy(bool) & available(co)
    return m


def paired_rows(co):
    from readout_sensitivity import paired_rows as original
    subjects, (ca, ia), (cb, ib) = original(co)
    keep = available(ca)[ia] & available(cb)[ib]
    return np.asarray(subjects)[keep].tolist(), (ca, ia[keep]), (cb, ib[keep])


def copy_json_tree(src, dest, omit=lambda p: False):
    for p in src.rglob('*.json'):
        rel = p.relative_to(src)
        if omit(rel):
            continue
        target = dest/rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)


def initialize():
    """Freeze task populations and stage unchanged results, never overwrite parents."""
    records, files = [], {}
    for co in sf.COHORTS:
        m = metadata(co)
        path = sf.RUN/'matches'/(co+'.json')
        files[str(path)] = sha(path)
        if co == 'nacc_repeated':
            continue
        for task, col, binary, cov in sf.task_specs(co, sf.labels(co)):
            tab = sf.labels(co)
            keep = common_mask(tab, co) & tab[col].notna().to_numpy()
            if cov:
                keep &= tab[cov].notna().all(axis=1).to_numpy()
            ix = np.flatnonzero(keep)
            records.append(dict(cohort=co, task=task, n=len(ix),
                scan_ids=m.iloc[ix].scan.tolist(), subjects=m.iloc[ix].subject.tolist()))
    for co in ('adni', 'nacc_repeated', 'act', 'cogte', 'breathe', 'psopt', 'hrv'):
        subjects, (ca, ia), (cb, ib) = paired_rows(co)
        records.append(dict(cohort=co, task='identification', n=len(subjects), subjects=subjects,
            scan_ids_a=metadata(ca).iloc[ia].scan.tolist(), scan_ids_b=metadata(cb).iloc[ib].scan.tolist()))
    save(RUN/'fm_matched_v3/intake.json', read(ROOT/'runs/fm_matched_v3/intake.json'))
    for directory in ('inference', 'fusion'):
        copy_json_tree(PARENT_PAPER/directory, PAPER/directory,
                       lambda p: 'cogte' in str(p))
    for version, dirs in [('demographic_benchmark_v1', ('evaluation',)),
                          ('readout_sensitivity_v1', ('cognition', 'identification')),
                          ('paper_robustness_v1', ('evaluation', 'weighting'))]:
        for directory in dirs:
            def omit(p):
                if version == 'readout_sensitivity_v1' and directory == 'identification':
                    return p.name in ('cogte.json', 'psopt.json', 'complete.json')
                return 'cogte' in str(p) or p.name == 'complete.json'
            copy_json_tree(ROOT/'runs'/version/directory, RUN/version/directory, omit)
    for version in ('paper_completion_v1', 'paper_robustness_v1'):
        save(RUN/version/'input_audit.json', read(ROOT/'runs'/version/'input_audit.json'))
    for name in ('spatial_audit.json', 'pretraining_audit.json'):
        save(RUN/'paper_robustness_v1'/name, read(ROOT/'runs/paper_robustness_v1'/name))
    # Robustness keeps its own additional complete-case/duration restrictions.
    robust = RUN/'paper_robustness_v1'
    for p in (ROOT/'runs/paper_robustness_v1/labels').glob('*.csv'):
        dest = robust/'labels'/p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): shutil.copyfile(p, dest)
    audit = read(robust/'input_audit.json')
    for p in (robust/'labels').glob('*.csv'):
        audit['files_sha256'][str(p)] = sha(p)
    save(robust/'input_audit.json', audit)
    for model in MODELS:
        for suffix in ('.npy', '.json'):
            src = ROOT/'runs/paper_robustness_v1/short_features'/model/('cogte'+suffix)
            dest = robust/'short_features'/model/src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists(): dest.symlink_to(src)
    save(RUN/'population_manifest.json', dict(status='complete', rows=records,
        sf100_match_sha256=files, policy='Same available scans within each endpoint; task-specific labels and clinical covariates retained. No extra QC.',
        excluded_scans=[dict(cohort=co, **r) for co in sf.COHORTS for r in sf.matches(co) if not r['available']],
        scope='Current manuscript panels: local cognition, frozen/SSL demographics, direct axes, existing frozen LOSO, readout identification, fusion and robustness.',
        not_added=['SF100/GFM window physiology', 'SF100/GFM LOSO (recommended, not yet requested)', 'MEM/EF-supervised age/sex'],
        archived_analyses='Older superseded queues and exploratory harmonization grids are not promoted to this release.',
        automatic_shutdown=False))


def local_rows(state):
    co='cogte'; m=metadata(co); keep=common_mask(m, co)
    f=np.load(pc.feature_dir(state)/(co+'.npy')); rows=[]
    for endpoint in endpoints(co):
        y=m[endpoint].to_numpy(float); ix=np.flatnonzero(keep & np.isfinite(y))
        g=m.iloc[ix].subject.to_numpy()
        rows.append(pc.row(co, endpoint+'_baseline', 'FM', m, ix, y[ix], oof(f[ix],y[ix],g)))
        clinical=pc.clinical_features(m,co); y=m['follow_'+endpoint].to_numpy(float)
        ix=np.flatnonzero(keep & np.isfinite(y) & np.isfinite(clinical).all(1));g=m.iloc[ix].subject.to_numpy()
        for label,blocks,dims in [('clinical',[clinical[ix]],[clinical.shape[1]]),
                                  ('clinical+FM',[clinical[ix],f[ix]],[clinical.shape[1],5])]:
            p=np.mean([oof_predictions(blocks,dims,y[ix],s,g) for s in (0,1,2)],axis=0)
            rows.append(pc.row(co,endpoint+'_followup',label,m,ix,y[ix],p))
    return rows


def canonical(state):
    name=pc.tag(*state); parent=PARENT_PAPER/'canonical'/(name+'.json'); d=read(parent)
    pc.verify_input_files([pc.feature_dir(state)/'cogte.npy'])
    d['rows']=[r for r in d['rows'] if r['cohort']!='cogte']+local_rows(state)
    # The target labels/scans never fitted source axes: fixed predictions can be rescored.
    valid=set(metadata('cogte').loc[common_mask(metadata('cogte'),'cogte'),'scan'])
    for r in d['axes']:
        if r['cohort']!='cogte': continue
        ix=np.flatnonzero([s in valid for s in r['scan_ids']])
        for key in ('scan_ids','subjects','y','prediction'): r[key]=np.asarray(r[key])[ix].tolist()
        r.update(n=len(ix),score=score(r['y'],r['prediction'],False))
    d.update(sample_alignment='Local probes refitted; fixed source-axis predictions rescored on common scans.',
             parent_sha256=sha(parent),population_manifest_sha256=sha(RUN/'population_manifest.json'))
    save(PAPER/'canonical'/(name+'.json'),d)


def baseline(atlas,method):
    from fm_matched_v3 import evaluate as ev
    ev.RUN=RUN/'fm_matched_v3'; ev.metadata=common_metadata
    ev.local('none','brainjepa' if atlas=='sch450' else 'brainlm13','frozen','none',0,'cogte',method)


def demographics(state):
    m=pd.read_csv(PARENT_DEMO/'labels/cogte.csv');f=np.load(demo.source_features(state)/'cogte.npy')
    name=pc.tag(*state)
    for task,col in [('AGE','age'),('SEX','sex')]:
        parent=PARENT_DEMO/'evaluation'/name/('cogte_'+task+'.json'); d=read(parent)
        ix=np.flatnonzero(m.primary_include.to_numpy(bool)&available('cogte')&m[col].notna().to_numpy())
        y=m.iloc[ix][col].to_numpy(float);g=m.iloc[ix].subject.astype(str).to_numpy();pred=[];audits=[]
        for seed in (0,1,2):
            prepared=prepare_nested([f[ix]],[100],y,g,seed)
            pred.append(predict_prepared(prepared,y,binary=task=='SEX'))
            audits.append(dict(seed=seed,folds=[dict(train_subjects=g[tr],test_subjects=g[te],inner_train_subjects=g[it],inner_validation_subjects=g[iv]) for tr,te,_,it,iv,_ in prepared]))
        pp=np.mean(pred,axis=0); sm=[demo.metrics(task,y,p) for p in pred]
        d.update(n=len(ix),n_subjects=len(set(g)),scan_ids=m.iloc[ix].scan.tolist(),subjects=g,y=y,prediction=pp,
                 predictions_by_probe_seed=pred,metrics=demo.metrics(task,y,pp),metrics_by_probe_seed=sm,
                 mean_probe_seed_metrics={k:np.mean([v[k] for v in sm]) for k in sm[0]},split_audit=audits,
                 sample_alignment='Refitted from cached embeddings on common scans',parent_sha256=sha(parent))
        d['signature']['population_manifest_sha256']=sha(RUN/'population_manifest.json')
        save(RUN/'demographic_benchmark_v1/evaluation'/name/parent.name,d)
    save(RUN/'demographic_done'/(name+'.json'),dict(status='complete'))


def resolved_read(path):
    p=Path(path)
    for version in ('fm_matched_v3','demographic_benchmark_v1'):
        base=ROOT/'runs'/version
        if p.is_relative_to(base):
            dest=RUN/version/p.relative_to(base)
            if dest.exists(): return read(dest)
            # A missing affected refit must not silently fall back to its parent.
            if 'cogte' in p.name: raise FileNotFoundError(dest)
    return read(p)


def cached_infer(key,contrasts):
    """Preserve unchanged conditional statistics; refit-dependent rows get fresh tests."""
    parent=PARENT_PAPER/'inference'/(key+'.json')
    old=read(parent)['rows'] if parent.exists() else []
    def identity(r): return {k:v for k,v in r.items() if k not in ('y','a','b','subjects','statistics')}
    keys=[json.dumps(identity(r),sort_keys=True) for r in old]
    counts=Counter(keys)
    # Legacy identities omitted analysis method. Two different predictors can
    # therefore share a key (notably HRV recall FM-only and clinical+FM). Never
    # overwrite one with the other or reuse ambiguous statistics.
    lookup={key:r for key,r in zip(keys,old) if counts[key]==1}; results=[]
    for c in contrasts:
        ident=identity(c); idx=json.dumps(ident,sort_keys=True)
        if c['cohort']!='cogte' and idx in lookup:
            results.append(lookup[idx]); continue
        seed=int(hashlib.sha256(idx.encode()).hexdigest()[:8],16)
        results.append(dict(**ident,statistics=pc.paired(c['y'],c['a'],c['b'],c['subjects'],c['metric'],seed=seed)))
    save(PAPER/'inference'/(key+'.json'),dict(status='complete',rows=results,
        sample_alignment='Changed CogTE contrasts recomputed; unaffected conditional statistics reused. Holm recalculated at release.'))


def configure_paper():
    pc.RUN=PAPER; pc.primary_mask=common_mask; pc.read=resolved_read; pc.infer_bundle=cached_infer


def inference(model,kind,source=None,objective=None,target=None):
    configure_paper()
    if kind=='frozen':pc.frozen_inference(model)
    elif kind=='demographics':pc.demographic_inference(model)
    else:pc.adaptation_inference(model,source,objective,target)
    key=kind+'_'+model if kind in ('frozen','demographics') else '_'.join(['adaptation',model,source,objective,target])
    save(RUN/'inference_done'/(key+'.json'),dict(status='complete'))


def loso(model,method):
    from fm_readout_v4 import supplement as su
    original=su.representation
    def features(co,mo,me,args):
        if me=='FM':return np.load(pc.feature_dir(args)/(co+'.npy'))
        return original(co,mo,me,args)
    su.RUN=RUN/'fm_readout_v4';su.metadata=common_metadata;su.representation=features
    su.loso('none',model,'frozen','none',0,method)


def identification(model):
    import readout_sensitivity as rs
    original_features=rs.features
    old_run=rs.RUN
    def features(mo,co):
        current=rs.RUN;rs.RUN=old_run
        try:return original_features(mo,co)
        finally:rs.RUN=current
    original_pairs=rs.paired_rows
    def pairs(co):
        subjects,(ca,ia),(cb,ib)=original_pairs(co)
        keep=available(ca)[ia]&available(cb)[ib]
        return np.asarray(subjects)[keep].tolist(),(ca,ia[keep]),(cb,ib[keep])
    rs.RUN=RUN/'readout_sensitivity_v1';rs.features=features;rs.paired_rows=pairs
    rs.identification(model)


def robustness(action,model=None,atlas=None,device='cpu'):
    import paper_robustness as pr
    pr.RUN=RUN/'paper_robustness_v1';pr.primary_mask=common_mask
    if action=='projections':pr.prepare_fc(atlas,'cogte',device)
    else:pr.evaluate(model,'cogte')


def report():
    configure_paper();pc.report()
    demo.RUN=RUN/'demographic_benchmark_v1'
    demo.report()
    import readout_sensitivity as rs
    rs.RUN=RUN/'readout_sensitivity_v1';rs.report()
    import paper_robustness as pr
    pr.RUN=RUN/'paper_robustness_v1';pr.report()
    # Verify exact scan membership, not only equal sample sizes.
    for p in (PAPER/'canonical').glob('*.json'):
        for r in read(p)['rows']:
            if r['cohort']!='cogte':continue
            task=r['task'].replace('_baseline','')
            ref=read(sf.RUN/'evaluation/cogte'/(task+'.json'))
            if r['scan_ids']!=ref['scan_ids'] or not np.array_equal(r['y'],ref['y']):
                raise ValueError('Final sample mismatch: '+str(p)+' '+task)
    for model in MODELS:
        for p in (demo.RUN/'evaluation'/(model+'_frozen')).glob('cogte_*.json'):
            d=read(p);ref=read(sf.RUN/'evaluation/cogte'/(d['task']+'.json'))
            assert d['scan_ids']==ref['scan_ids'] and np.array_equal(d['y'],ref['y'])
    save(RUN/'report.json',dict(status='complete',population_manifest_sha256=sha(RUN/'population_manifest.json'),
        scope=read(RUN/'population_manifest.json')['scope'],sample_verification='Exact scan IDs/outcomes match the SF100 task panel for every changed cognitive state and frozen demographics.',
        encoder_training_performed=False,automatic_shutdown=False))
