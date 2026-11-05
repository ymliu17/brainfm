"""CPU-only, common frozen-window evaluation of ECG and PPG cohorts."""
from pathlib import Path
import json
import re
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'scripts'))
from fm_matched_v3.data import DATA, dataset, save, sha
from fm_matched_v3.supplement import window_dir, window_signature
from fm_matrix.models import MODELS
from fm_matched_v3 import physiology as engine
from benchmark_evaluation import grouped_splits
from paper_paired_stats import paired, holm
from transfer_baseline import fc_feature, amp_feature

RUN = ROOT / 'runs/physiology_frozen_crosscohort_v2_no_usable'
COHORTS = ('psopt', 'hrv', 'breathe')
ECG = {'psopt': 'pSOPT_MRI_MeanHR_windows_27.5s.csv',
       'breathe': 'BREATHE_MRI_MeanHR_windows_20.2s.csv'}
SEEDS = (0, 1, 2)


def key(s):
    return re.search(r'(\d+)', str(s)).group(1).lstrip('0')


def metadata(cohort):
    names = ('hrv_pre', 'hrv_post') if cohort == 'hrv' else (cohort,)
    tables = []
    for name in names:
        m = pd.read_csv(DATA / f'{name}.csv')
        m['dataset'] = name
        m['matrix_index'] = np.arange(len(m))
        m['key'] = m.subject if cohort == 'hrv' else m.subject.map(key)
        tables.append(m)
    m = pd.concat(tables, ignore_index=True)
    assert not m.duplicated(['key', 'visit']).any()
    return m


def hrv_time_rows(w):
    valid = w.ScanAligned.eq(True) & w.FullWindow.eq(True) & w.TimeReference.eq('scan') & w.Task.eq('rest') & w.TRSec.eq(2.4)
    valid &= w.TRsPerWindow.eq(20) & np.isclose(w.WindowLengthSec,48) & np.isclose(w.StepSec,48)
    valid &= w.VolumeStartIdx.eq(w.WindowIdx*20) & w.VolumeEndIdxExclusive.eq((w.WindowIdx+1)*20)
    valid &= np.isclose(w.ScanStartSec,w.VolumeStartIdx*2.4) & np.isclose(w.ScanEndSec,w.VolumeEndIdxExclusive*2.4)
    return valid


def select_windows(cohort):
    m = metadata(cohort)
    if cohort == 'hrv':
        # Rejoin raw PPG rows: never inherit the old Usable-filtered HRV sample.
        source = ROOT / 'data/HRV_biofeedback/label/HRV_biofeedback_PPG_HRV_windows48s_full.csv'
        w = pd.read_csv(source)
        w = w[hrv_time_rows(w) & w.WindowIdx.between(0, 7)].copy()
        w['key'] = w.Subject
        w['visit'] = w.Timepoint
        w['rate'] = w.HRV_MeanHR
        width = 48.
    else:
        source = ROOT / 'data/rct' / cohort / ECG[cohort]
        w = pd.read_csv(source)
        if 'Run' in w:
            w = w[w.Run.eq('rest-run01')]
        w = w[w.FullWindow.eq(True) & w.MeanHR.between(40, 120)
              & w.WindowIdx.between(0, 7)].copy()
        w['key'] = w.Subject.map(key)
        w['visit'] = w.Timepoint
        w['rate'] = w.MeanHR
        width = 27.5 if cohort == 'psopt' else 20.2
        np.testing.assert_allclose(w.fMRIWindowStartSec, w.WindowIdx * width, atol=1e-6)
        np.testing.assert_allclose(w.fMRIWindowEndSec, (w.WindowIdx + 1) * width, atol=1e-6)
    assert not w.duplicated(['key', 'visit', 'WindowIdx']).any()
    cols = ['key', 'visit', 'dataset', 'matrix_index', 'subject', 'native_frames']
    w = w.drop(columns=[c for c in cols[2:] if c in w])
    w = w.merge(m[cols], on=['key', 'visit'], how='inner', validate='many_to_one')
    assert (w.native_frames >= 160).all()
    # All methods use the intersection of exported scan indices.
    for model in MODELS:
        keep = set()
        f = window_dir('none', model, 'frozen', 'none', 0)
        for name in m.dataset.unique():
            keep.update((name, int(i)) for i in np.load(f / f'{name}_window_indices.npy'))
        w = w[[v in keep for v in zip(w.dataset, w.matrix_index)]]
    w['Subject'] = w.subject
    w['Timepoint'] = w.visit
    w['scan'] = w.Subject + '_' + w.visit
    w = w[np.isfinite(w.rate)].copy()
    w = w[w.groupby('scan').rate.transform('size') >= 6]
    w = w.sort_values(['scan', 'WindowIdx']).reset_index(drop=True)
    assert len(w) and w.Subject.nunique() >= 5
    assert (w.groupby('scan').size().between(6, 8)).all()
    return w, source, width


def audit():
    RUN.mkdir(exist_ok=True)
    if (RUN / 'protocol.json').exists():
        verify(); return
    files = {}
    def record(p):
        files[str(p.relative_to(ROOT))] = sha(p)
    for p in [Path(__file__), ROOT / 'scripts/run_ecg_frozen_no_usable.py',
              ROOT / 'src/fm_matched_v3/physiology.py', ROOT / 'src/paper_paired_stats.py',
              ROOT / 'data/rct/breathe/meta.json', ROOT / 'runs/fm_matched_v3/intake.json']:
        record(p)
    assert json.loads((ROOT / 'data/rct/breathe/meta.json').read_text())['native_tr'] == 1.01
    for model in MODELS:
        args = ('none', model, 'frozen', 'none', 0)
        f = window_dir(*args)
        d = json.loads((f / 'complete.json').read_text())
        assert d['status'] == 'complete' and d['signature'] == window_signature(*args)
        record(f / 'complete.json')
        for name in ['psopt', 'breathe', 'hrv_pre', 'hrv_post']:
            for suffix in ['windows', 'window_indices']:
                record(f / f'{name}_{suffix}.npy')
            z = np.load(f / f'{name}_windows.npy', mmap_mode='r')
            ix = np.load(f / f'{name}_window_indices.npy')
            assert z.shape[:2] == (len(ix), 8) and len(set(ix)) == len(ix)
            assert np.isfinite(z).all()
    counts = {}
    for co in COHORTS:
        w, source, width = select_windows(co)
        record(source)
        for name in w.dataset.unique():
            record(DATA / f'{name}.csv')
        save(RUN / co / 'intake.json', dict(status='complete', width_seconds=width,
             subjects=w.Subject.nunique(), scans=w.scan.nunique(), windows=len(w),
             window_counts=w.groupby('scan').size().value_counts().to_dict(),
             scope='common frozen-window evaluation without physiological usability marks'))
        w.to_csv(RUN / co / 'windows.csv', index=False)
        counts[co] = json.loads((RUN / co / 'intake.json').read_text())
    save(RUN / 'protocol.json', dict(status='specified_before_new_scores', files_sha256=files, counts=counts,
        models=list(MODELS), states='frozen only; no adapter fitting or encoder extraction',
        readout='Cached final-layer ROI/content-weighted temporal-token means; eight 20-frame windows; full offline context over first160 real frames; BrainLM zero-padded to200, JEPA160.',
        sample='User-requested removal of all physiological Usable_HR/Usable/RunUsable/SignalQCUsable flags in all3cohorts. Raw ECG/PPG tables are rejoined, without inherited filtered HRV manifests. Retain common model/atlas scans, full/time-aligned first8windows, finite rate, >=6windows/scan; ECG40–120bpm and rest-run01 BREATHE unchanged. No cognitive-label eligibility or RMSSD requirement.',
        endpoint='Mean heart rate (ECG) / pulse rate (PPG); no pooling of cardiac modalities or cohorts. No new HRV/PRV variability or post-training endpoints.',
        dynamic='Demean outcomes, features and predictions over retained windows within each scan. Window-position-by-visit nuisance in every arm. Common 5-fold participant-grouped outer CV and grouped20% inner validation, seeds0/1/2.',
        scan_mean='Separate mean rate over retained windows; imaging features averaged over the same retained windows. Visit-only nuisance shared across cohorts; not a whole-recording endpoint or a clinical incremental-value test.',
        controls='Same-atlas/same20-frame FC and regional amplitude; all learned scaler/PCA steps refitted in training folds. Up to10 imaging PCs, from a shared training-fitted up-to20 PC basis. Ridge C=.001,.01,.1,1,10 selected by inner Spearman.',
        inference='5000 participant-cluster bootstrap replicates;20000 two-sided participant-block prediction swaps, no model refitting. Spearman weights retained windows/scans equally, clusters whole participants. Seed-average OOF predictions.',
        families='Three dynamic families: FM versus FC, AMP or nuisance, each across all4models and3cohorts (12 contrasts). One exploratory scan-mean family of36contrasts. All3cohorts retained in families; no data-dependent family selection.',
        temporal_null='No new circular-shift null tests; paired tests address relative predictive performance, not absolute physiological decoding.',
        provenance='BREATHE TR1.01s recorded from user-provided BIDS sidecar in local meta.json; pSOPT TR1.375s; HRV TR2.4s. Original scanner trigger synchronization not independently reaudited.',
        gpu_required=False, automatic_shutdown=False))


def verify():
    d = json.loads((RUN / 'protocol.json').read_text())
    for p, expected in d['files_sha256'].items():
        if sha(ROOT / p) != expected:
            raise ValueError('Registered input changed: ' + p)
    return d


def prepare(co):
    out = RUN / co
    if (out / 'prepared.json').exists():
        return
    w = pd.read_csv(out / 'windows.csv')
    scans = w.drop_duplicates('scan').reset_index(drop=True)
    scans.to_csv(out / 'scans.csv', index=False)
    for atlas, model in [('a424', 'brainlm13'), ('sch450', 'brainjepa')]:
        xs = {name: dataset(name, model)[0] for name in w.dataset.unique()}
        x = torch.stack([xs[r.dataset][r.matrix_index, :, int(r.WindowIdx)*20:(int(r.WindowIdx)+1)*20]
                         for r in w.itertuples()]).numpy()
        for kind, func in [('FC', fc_feature), ('AMP', amp_feature)]:
            a = np.asarray(func(x, None), dtype=np.float32)
            assert np.isfinite(a).all()
            np.save(out / f'{atlas}_{kind}.npy', a)
    for model in MODELS:
        f = window_dir('none', model, 'frozen', 'none', 0)
        lookup = {}
        for name in w.dataset.unique():
            z = np.load(f / f'{name}_windows.npy')
            ix = np.load(f / f'{name}_window_indices.npy')
            lookup.update({(name, int(i)): a for i, a in zip(ix, z)})
        z = np.stack([lookup[r.dataset, r.matrix_index][int(r.WindowIdx)] for r in w.itertuples()])
        assert np.isfinite(z).all()
        np.save(out / f'{model}_FM.npy', z)
    for scope, m in [('dynamic', w), ('scan_mean', scans)]:
        g = m.Subject.to_numpy()
        for seed in SEEDS:
            folds = []
            for tr, te in grouped_splits(np.zeros(len(g)), g, seed):
                a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(tr, groups=g[tr]))
                assert not set(g[tr]) & set(g[te])
                assert not set(g[tr[a]]) & set(g[tr[b]])
                folds.append(dict(train=tr, test=te, inner_train=tr[a], inner_validation=tr[b]))
            save(out / f'{scope}_splits_s{seed}.json', folds)
    # Helper cache identities use this protocol hash; no parent files are modified.
    save(out / 'protocol.json', dict(parent_sha256=sha(RUN / 'protocol.json'), cohort=co))
    save(out / 'prepared.json', dict(status='complete', artifacts={p.name:sha(p) for p in out.glob('*.npy')}))


def mean_scans(x, scans):
    return np.stack([x[scans == s].mean(0) for s in pd.unique(scans)])


def evaluate(co):
    out = RUN / co
    if (out / 'evaluation.json').exists():
        return co
    prepare(co)
    engine.RUN = out
    w = pd.read_csv(out / 'windows.csv')
    scans = w.scan.to_numpy()
    sm = pd.read_csv(out / 'scans.csv')
    rows = []
    for scope in ['dynamic', 'scan_mean']:
        dynamic = scope == 'dynamic'
        meta = w if dynamic else sm
        nuisance = pd.get_dummies((w.Timepoint + '_' + w.WindowIdx.astype(str)) if dynamic else sm.Timepoint, dtype=float).to_numpy()
        if dynamic:
            nuisance = engine.center_scans(nuisance, scans)
        nfit = engine.cached_transforms(nuisance, scope, nuisance.shape[1])
        y = engine.center_scans(w.rate.to_numpy(), scans) if dynamic else mean_scans(w.rate.to_numpy(), scans)
        predictions = {'nuisance':engine.predict_many(nfit, y, scope, meta.scan.to_numpy())[:, 0]}
        for model in MODELS:
            atlas = 'sch450' if model == 'brainjepa' else 'a424'
            for kind in ['FC', 'AMP', 'FM']:
                name = model + '_FM' if kind == 'FM' else atlas + '_' + kind
                if name in predictions:
                    continue
                x = np.load(out / f'{name}.npy')
                x = engine.center_scans(x, scans) if dynamic else mean_scans(x, scans)
                fit = engine.cached_transforms(x, scope, 10)
                combined = [[tuple(np.hstack([nfit[s][i][j], fit[s][i][j]]) for j in range(4))
                             for i in range(5)] for s in range(3)]
                predictions[name] = engine.predict_many(combined, y, scope, meta.scan.to_numpy())[:, 0]
                print(co, scope, name, engine.summarize(y, predictions[name]), flush=True)
        row = dict(scope=scope, y=y, subjects=meta.Subject.to_numpy(), scans=meta.scan.to_numpy(),
                   predictions=predictions, scores={k:engine.summarize(y, v) for k,v in predictions.items()})
        rows.append(row)
        save(out / 'evaluation.progress.json', dict(status='running', rows=rows))
    save(out / 'evaluation.json', dict(status='complete', cohort=co, rows=rows, protocol_sha256=sha(RUN / 'protocol.json')))
    return co


def infer(co, model):
    path = RUN / co / f'inference_{model}.json'
    if path.exists():
        return co + ':' + model
    data = json.loads((RUN / co / 'evaluation.json').read_text())
    atlas = 'sch450' if model == 'brainjepa' else 'a424'
    rows = []
    for row in data['rows']:
        for ref in ['nuisance', atlas+'_FC', atlas+'_AMP']:
            stat = paired(row['y'], row['predictions'][ref], row['predictions'][model+'_FM'], row['subjects'])
            stat['uncertainty'] = 'Conditional on fitted OOF predictions; retained-window/scan-weighted Spearman, participant-cluster resampling'
            family = 'scan_mean' if row['scope'] == 'scan_mean' else 'dynamic_' + ref.split('_')[-1]
            rows.append(dict(cohort=co, model=model, scope=row['scope'], reference=ref, candidate=model+'_FM', family=family, **stat))
    save(path, dict(status='complete', comparisons=rows))
    return co + ':' + model


def report():
    scores, comparisons = [], []
    for co in COHORTS:
        d = json.loads((RUN / co / 'evaluation.json').read_text())
        for r in d['rows']:
            for method, v in r['scores'].items():
                scores.append(dict(cohort=co, scope=r['scope'], method=method, n=len(r['y']),
                     subjects=len(set(r['subjects'])), scans=len(set(r['scans'])), **v))
        for model in MODELS:
            comparisons.extend(json.loads((RUN / co / f'inference_{model}.json').read_text())['comparisons'])
    d = pd.DataFrame(comparisons)
    assert len(d) == 72
    for family, g in d.groupby('family'):
        d.loc[g.index, 'holm_p'] = holm(g.permutation_p.tolist())
    pd.DataFrame(scores).to_csv(RUN / 'scores.csv', index=False)
    d.to_csv(RUN / 'paired_comparisons.csv', index=False)
    save(RUN / 'report.json', dict(status='complete', cohorts=list(COHORTS), scores=len(scores), contrasts=len(d),
         positive=int(((d.holm_p<.05)&(d.delta>0)).sum()), negative=int(((d.holm_p<.05)&(d.delta<0)).sum()),
         families=d.groupby('family').size().to_dict(), automatic_shutdown=False))
