#!/usr/bin/env python3
"""Nine-model extension of the registered BrainLM13 Re-ID controls.

Additive artifacts only. No post-training, manuscript edits, or shutdown.
GPU extraction and CPU comparisons run independently and resume from artifacts.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from fm_matrix.data import save, sha
import run_connectivity_readouts as frozen
import run_identity_controls as old
import readout_sensitivity as rs
from paper_paired_stats import holm

RUN = ROOT/'runs/nine_model_identity_controls_v1'
OUT = ROOT/'results/generated/nine_model_identity_controls_v1'
LOG = ROOT/'logs/nine_model_identity_controls_v1'
MODELS = frozen.MODELS
COHORTS = old.COHORTS
SEEDS = old.SEEDS
MODES = old.MODES
METRICS = old.METRICS


def read(p): return json.loads(Path(p).read_text())
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def pair(co): return read(old.RUN/'pairs'/f'{co}.json')
def random_folder(model, seed, co): return RUN/'features'/model/str(seed)/co
def complete(p): return p.exists() and read(p).get('status') == 'complete'


def register():
    sources = {}
    for co in COHORTS:
        p = old.RUN/'pairs'/f'{co}.json'
        sources[str(p)] = sha(p)
        for model in MODELS:
            q = frozen.RUN/'identity'/model/f'{co}.json'
            assert read(q)['subjects'] == pair(co)['subjects']
            sources[str(q)] = sha(q)
    for p in [Path(__file__), ROOT/'src/readout_control_encoders.py', Path(old.__file__),
              Path(rs.__file__), ROOT/'src/unified_supplements.py', old.FD]:
        sources[str(p)] = sha(p)
    d = dict(version=1, models=MODELS, cohorts=COHORTS, random_seeds=SEEDS,
        scope='All nine frozen configurations: primary and normalized final-layer spatial embedding connectivity only. Seven cohorts, identical participant pairs. Other readouts and cognition already evaluated; no duplicate probe fitting.',
        initialization='Fresh official constructors for every learned encoder component. Preserve fixed atlas, gradient and positional priors. fMRI-LM randomizes both tokenizer and language backbone; this does not isolate image versus language pretraining. Five seeds fixed across visits/cohorts. BrainLM13 uses verified existing five-seed artifacts.',
        layer='Both readouts use the same final encoder output, including official final normalization. Spatial units remain model-specific: ROI, image patch row, or network; not all are ROI embeddings.',
        extraction='Same registered input tensors and missing-input masks; eval, no random masking, bfloat16 autocast. Preserve canonical batch4 for fMRI-LM (including canonical row positions and last-batch size), batch1 otherwise. Pretrained cache parity required for each configuration before random extraction.',
        summaries='ROI standard deviation (ddof0) and lag1 autocorrelation from real samples in model-matched time windows, before model-specific per-ROI scaling. Original four use canonical inputs. New models use verified original source signals, with TR2 resampling for Semantoks/fMRI-LM. Cerebellar missing rows and padding excluded. These controls may retain amplitude removed by model normalization.',
        statistics='5000 participant bootstraps and20000 two-sided paired query-score swaps conditional on fixed gallery. Random baseline averages per-participant scores across seeds, not embeddings or similarity matrices. Native dimensions and data-independent Gaussian128 projections with seeds0/1/2.',
        contrasts='Signal panel: primary and connectivity versus ROIstd/lag1, plus connectivity versus same-input-window FC. Random panel: pretrained versus random at both readouts, and pretraining-by-readout difference-in-differences. All seed-specific scores retained.',
        correction='Separate Holm families for signal, random and motion panels, each split by dimensionality mode and metric. Within a family include every model, cohort and registered contrast (motion: every model/split/contrast). Adjusted values released only after whole panel completion. Existing families/results remain unchanged.',
        motion='Verified pSOPT paired meanFD only. Three fixed20% calibration/80% evaluation splits from original13M control. Fit log1pFD OLS residualization in calibration participants. Compare adjusted connectivity with adjusted primary/FC/random and unadjusted connectivity on the same evaluation gallery. Motion-only score descriptive. No assumed FD coverage in other cohorts.',
        gpu_order='One full seed across configurations before later seeds; pretrained parity first. Two GPU workers, four CPU workers; no training.',
        source_sha256=sources, automatic_shutdown=False, manuscript_modified=False)
    p = RUN/'protocol.json'
    from fm_matrix.data import clean
    if p.exists():
        assert read(p) == clean(d), 'Registered protocol changed'
    else:
        save(p, d)
    return d


def verify():
    d = read(RUN/'protocol.json')
    for p, digest in d['source_sha256'].items():
        assert sha(p) == digest, p


def infer(net, x, device):
    import torch
    with torch.inference_mode(), torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
        return net(x.to(device))


def smoke(model, device):
    import torch
    from readout_control_encoders import ControlEncoder, inputs
    torch.set_num_threads(2)
    p = pair('psopt')
    x, m = inputs(model, p['a']['cohort'])
    ix = [p[s]['indices'][0] for s in ('a', 'b')]
    xx = x[ix]
    net = ControlEncoder(model)
    prehash = net.fingerprint()
    net.to(device)
    rows = []
    start = time.time()
    batch = 4 if model.startswith('fmrilm') else 1
    values = []
    for index in ix:
        start_batch = index//batch*batch
        z = infer(net, x[start_batch:start_batch+batch], device)
        values.append({k: v[index-start_batch:index-start_batch+1] for k, v in z.items()})
    for name in ('native', 'connectivity'):
        cached = np.load(frozen.RUN/'features'/model/'psopt'/f'{name}.npy', mmap_mode='r')[ix]
        actual = np.concatenate([v[name] for v in values])
        rmse = float(np.sqrt(np.mean((actual-cached)**2))/max(np.sqrt(np.mean(cached**2)), 1e-12))
        cosine = np.sum(actual*cached, axis=1)/np.maximum(np.linalg.norm(actual, axis=1)*np.linalg.norm(cached, axis=1), 1e-12)
        assert rmse < .005 and cosine.min() > .9999, (model, name, rmse, cosine)
        rows.append(dict(readout=name, relative_rmse=rmse, min_cosine=float(cosine.min()), dimensions=actual.shape[1]))
    del net
    torch.cuda.empty_cache()
    net = ControlEncoder(model, SEEDS[0])
    h = net.fingerprint()
    assert h != prehash
    net.to(device)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    test_batch = x[:batch]
    a = infer(net, test_batch, device)
    torch.cuda.synchronize()
    seconds = (time.time()-start)/len(test_batch)
    b = infer(net, test_batch, device)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        assert np.isfinite(a[key]).all()
    peak = torch.cuda.max_memory_allocated()
    del net
    torch.cuda.empty_cache()
    # Constructor determinism is tested without any learned checkpoint load.
    other = ControlEncoder(model, SEEDS[0])
    assert other.fingerprint() == h
    del other
    save(RUN/'smoke'/f'{model}.json', dict(status='complete', model=model, device=device,
        pretrained_parity=rows, random_hash=h, pretrained_hash=prehash, deterministic=True,
        same_seed_reproduced=True, batch_size=batch, random_seconds_per_scan=seconds, peak_cuda_bytes=peak,
        backend_sha256=sha(ROOT/'src/readout_control_encoders.py')))
    print('smoke passed', model, seconds, peak/2**30, flush=True)


def random_extract(model, seed, device):
    import torch
    from readout_control_encoders import ControlEncoder, inputs
    torch.set_num_threads(2)
    assert complete(RUN/'smoke'/f'{model}.json')
    net = ControlEncoder(model, seed)
    digest = net.fingerprint()
    net.to(device)
    start = time.time()
    total_done = 0
    for co in COHORTS:
        dest = random_folder(model, seed, co)
        if complete(dest/'complete.json'):
            continue
        arrays = {}
        p = pair(co)
        for side in ('a', 'b'):
            item = p[side]
            x, m = inputs(model, item['cohort'])
            assert m.iloc[item['indices']].scan.tolist() == item['scans']
            vals = {k: [] for k in ('native', 'connectivity')}
            for j, ix in enumerate(item['indices']):
                chunk = dest/'chunks'/f'{side}_{j:04d}.npz'
                if chunk.exists():
                    with np.load(chunk) as z:
                        got = {k: z[k] for k in z.files}
                else:
                    batch = 4 if model.startswith('fmrilm') else 1
                    start_batch = ix//batch*batch
                    whole = infer(net, x[start_batch:start_batch+batch], device)
                    got = {k: v[ix-start_batch] for k, v in whole.items()}
                    if not all(np.isfinite(v).all() for v in got.values()):
                        raise ValueError('Nonfinite random readout')
                    # Cache paired scans in either visit from this canonical batch.
                    # This preserves the historical bf16 path without recomputing
                    # the same batch for an interleaved baseline/follow-up pair.
                    for other_side in ('a', 'b'):
                        other = p[other_side]
                        if other['cohort'] != item['cohort']: continue
                        for other_j, other_ix in enumerate(other['indices']):
                            if start_batch <= other_ix < start_batch+len(whole['native']):
                                target = dest/'chunks'/f'{other_side}_{other_j:04d}.npz'
                                if not target.exists():
                                    old.array_save(target, **{k: v[other_ix-start_batch] for k, v in whole.items()})
                for k, v in got.items():
                    vals[k].append(v)
                total_done += 1
                if j % 25 == 0:
                    save(RUN/'progress'/f'{model}_{seed}.json', dict(model=model, seed=seed,
                        cohort=co, side=side, current=j+1, side_total=len(item['indices']),
                        scans_done_this_job=total_done, seconds=time.time()-start, updated=utc()))
                    print(model, seed, co, side, j+1, len(item['indices']), flush=True)
            arrays.update({k+'_'+side: np.stack(v) for k, v in vals.items()})
            del x
        old.array_save(dest/'features.npz', **arrays)
        save(dest/'complete.json', dict(status='complete', model=model, seed=seed, cohort=co,
            encoder_sha256=digest, sha256=sha(dest/'features.npz'),
            pair_sha256=sha(old.RUN/'pairs'/f'{co}.json'), backend_sha256=sha(ROOT/'src/readout_control_encoders.py')))
        import shutil
        shutil.rmtree(dest/'chunks')


def random_arrays(model, seed, co):
    if model == 'brainlm13':
        return old.load_arrays(f'random{seed}', co)
    p = random_folder(model, seed, co)
    stamp = read(p/'complete.json')
    assert stamp['sha256'] == sha(p/'features.npz')
    assert stamp['pair_sha256'] == sha(old.RUN/'pairs'/f'{co}.json')
    with np.load(p/'features.npz') as z:
        return {k: z[k] for k in z.files}


def input_kind(model):
    return ('a424' if model.startswith('brainlm') else 'sch450' if model == 'brainjepa'
        else 'semantoks' if model == 'semantoks' else 'harmonix' if model == 'brainharmonix_f' else 'fmrilm')


def summaries(co):
    """Build all five input-family signal summaries once per cohort."""
    from fm_readout_v4.data import dataset, metadata
    from new_fm.data import source_rows
    from temporal import resample_time
    p = pair(co)
    result = {kind: {} for kind in ('a424', 'sch450', 'semantoks', 'fmrilm', 'harmonix')}
    sources = {}
    expected = read(ROOT/'runs/new_fm_v1/intake.json')['sources_sha256']
    order = np.array(read(ROOT/'data/derived/new_fm_v1/roi_mapping.json')['permutation'])
    for side in ('a', 'b'):
        item = p[side]
        co_input = item['cohort']
        for kind, model, width in [('a424', 'brainlm13', 200), ('sch450', 'brainjepa', 160)]:
            x, m = dataset(co_input, model)
            assert m.iloc[item['indices']].scan.tolist() == item['scans']
            vals = [old.input_summaries(x[ix].numpy(), min(int(m.iloc[ix].native_frames), width)) for ix in item['indices']]
            result[kind]['roi_std_'+side] = np.stack([v[0] for v in vals])
            result[kind]['roi_lag1_'+side] = np.stack([v[1] for v in vals])
            del x
        wanted = set(item['indices'])
        lookup = {}
        for ix, (x, tr, path) in enumerate(source_rows(co_input)):
            if ix not in wanted:
                continue
            path = str(path)
            if path not in sources:
                digest = sha(path)
                assert expected.get(path) == digest, 'Original source changed: '+path
                sources[path] = digest
            z = resample_time(x.float(), tr, 2.).numpy()
            lookup[ix] = {
                'semantoks': old.input_summaries(z[order], min(z.shape[1], 100)),
                'fmrilm': old.input_summaries(z[np.r_[order[400:], order[:400]]], min(z.shape[1], 160)),
                'harmonix': old.input_summaries(x[:400].numpy(), min(x.shape[1], 18*round(35.28/tr)))}
        assert wanted == set(lookup)
        for kind in ('semantoks', 'fmrilm', 'harmonix'):
            for j, name in enumerate(('roi_std', 'roi_lag1')):
                result[kind][name+'_'+side] = np.stack([lookup[ix][kind][j] for ix in item['indices']])
    for kind, arrays in result.items():
        dest = RUN/'summaries'/co/kind
        old.array_save(dest/'features.npz', **arrays)
        scores = old.score_arrays(arrays)
        save(dest/'complete.json', dict(status='complete', scores=scores, sha256=sha(dest/'features.npz'),
            pair_sha256=sha(old.RUN/'pairs'/f'{co}.json'), sources_sha256=sources))
    # Exact legacy parity for the existing13M standard-deviation/lag controls.
    legacy = old.load_arrays('controls', co)
    for k, value in result['a424'].items():
        np.testing.assert_array_equal(value, legacy[k])
    save(RUN/'summaries'/co/'complete.json', dict(status='complete', legacy_13M_parity=True))


def fc_record(model, co):
    base = ROOT/('runs/brainharmonix_f_v1/supplements' if model == 'brainharmonix_f' else 'runs/reviewer_completion_v1')
    return read(base/'identity'/model/f'{co}.json')


def compare(model, co, panel, mode, metric, ref, cand, name):
    path = RUN/'parts'/panel/model/co/mode/metric/f'{name}.json'
    if path.exists():
        return read(path)
    d = rs.paired_identification(ref, cand, frozen.seed('|'.join((model, co, panel, mode, metric, name))))
    d.update(model=model, cohort=co, panel=panel, mode=mode, metric=metric, comparison=name,
        family=f'{panel}::{mode}::{metric}')
    save(path, d)
    return d


def evaluate(model, co, panel):
    prior = read(frozen.RUN/'identity'/model/f'{co}.json')
    rows = []
    if panel == 'signal':
        controls = read(RUN/'summaries'/co/input_kind(model)/'complete.json')['scores']
        fcrec = fc_record(model, co)
        assert fcrec['subjects'] == prior['subjects']
        fc = {x['mode']: x['FC'] for x in fcrec['scores']}
        seeds = {}
    else:
        seeds = {str(seed): old.score_arrays(random_arrays(model, seed, co)) for seed in SEEDS}
    for mode in MODES:
        for metric in METRICS:
            key = 'per_subject_'+metric
            pre = {k: np.array(prior['scores'][mode][k][key]) for k in ('native', 'connectivity')}
            if panel == 'signal':
                contrasts = [(controls[mode][ref][key], pre[cand], cand+'_minus_'+ref)
                    for ref in ('roi_std', 'roi_lag1') for cand in ('native', 'connectivity')]
                contrasts.append((fc[mode][key], pre['connectivity'], 'connectivity_minus_input_FC'))
            else:
                rand = {k: np.mean([seeds[str(s)][mode][k][key] for s in SEEDS], axis=0) for k in pre}
                contrasts = [(rand[k], pre[k], 'pretrained_minus_random_'+k) for k in pre]
                contrasts.append((rand['connectivity']-rand['native'], pre['connectivity']-pre['native'],
                    'pretraining_by_readout_interaction'))
            for ref, cand, name in contrasts:
                rows.append(compare(model, co, panel, mode, metric, ref, cand, name))
    save(RUN/'evaluation'/panel/model/f'{co}.json', dict(status='complete', rows=rows,
        random_seed_scores=seeds, subjects=prior['subjects']))


def motion(model):
    from sklearn.model_selection import train_test_split
    from fm_readout_v4.data import metadata
    co = 'psopt'
    p = pair(co)
    fd = pd.read_csv(old.FD)
    assert not fd.duplicated(['subject', 'timepoint']).any()
    lookup = fd.set_index(['subject', 'timepoint']).mean_fd
    vals, arrays = {}, {}
    for side in ('a', 'b'):
        item = p[side]
        m = metadata(item['cohort']).iloc[item['indices']]
        vals[side] = np.array([lookup.get((str(s).removeprefix('sub-'), v), np.nan) for s, v in zip(m.subject, m.visit)])
        folder = frozen.RUN/'features'/model/item['cohort']
        stamp = read(folder/'complete.json')
        assert stamp['scan_ids'] == metadata(item['cohort']).scan.tolist()
        for name in ('native', 'connectivity'):
            path = folder/f'{name}.npy'
            assert sha(path) == stamp['features_sha256'][name]
            arrays[name+'_'+side] = np.load(path, mmap_mode='r')[item['indices']].copy()
        if model == 'brainharmonix_f':
            fcp = ROOT/'runs/brainharmonix_f_v1/supplements/input_fc'/model/f'{item["cohort"]}.npy'
        else:
            fcp = ROOT/'runs/reviewer_completion_v1/input_fc'/input_kind(model)/f'{item["cohort"]}.npy'
        stamp = read(fcp.with_suffix('.json'))
        assert sha(fcp) == stamp['sha256']
        assert stamp['scan_ids'] == metadata(item['cohort']).scan.tolist()
        arrays['FC_'+side] = np.load(fcp, mmap_mode='r')[item['indices']].copy()
    for seed in SEEDS:
        arrays.update({f'random{seed}_{k}': v for k, v in random_arrays(model, seed, co).items()})
    keep = np.isfinite(vals['a']) & np.isfinite(vals['b'])
    assert (vals['a'][keep] >= 0).all() and (vals['b'][keep] >= 0).all()
    fa, fb = [np.log1p(vals[s][keep]) for s in ('a', 'b')]
    arrays = {k: v[keep] for k, v in arrays.items()}
    rows, splits = [], []
    for seed in (0, 1, 2):
        cal, ev = train_test_split(np.arange(keep.sum()), train_size=.2, random_state=seed)
        xc = np.column_stack((np.ones(2*len(cal)), np.r_[fa[cal], fb[cal]]))
        xa, xb = [np.column_stack((np.ones(len(ev)), f[ev])) for f in (fa, fb)]
        adj, raw = {}, {}
        for name in sorted(k[:-2] for k in arrays if k.endswith('_a')):
            a, b = arrays[name+'_a'], arrays[name+'_b']
            beta = np.linalg.lstsq(xc, np.r_[a[cal], b[cal]], rcond=None)[0]
            adj[name+'_a'], adj[name+'_b'] = a[ev]-xa@beta, b[ev]-xb@beta
            raw[name+'_a'], raw[name+'_b'] = a[ev], b[ev]
        adjusted = old.score_arrays(adj)
        original = old.score_arrays({k: v for k, v in raw.items() if k.startswith('connectivity_')})
        for mode in MODES:
            for metric in METRICS:
                key = 'per_subject_'+metric
                cand = adjusted[mode]['connectivity'][key]
                rmean = np.mean([adjusted[mode][f'random{s}_connectivity'][key] for s in SEEDS], axis=0)
                for ref, name in [(adjusted[mode]['native'][key], 'connectivity_minus_primary'),
                                  (adjusted[mode]['FC'][key], 'connectivity_minus_FC'),
                                  (rmean, 'connectivity_minus_random'),
                                  (original[mode]['connectivity'][key], 'adjusted_minus_unadjusted')]:
                    rows.append(compare(model, co, 'motion', mode, metric, ref, cand, f'split{seed}_'+name))
        splits.append(dict(seed=seed, calibration_subjects=np.array(p['subjects'])[keep][cal],
            evaluation_subjects=np.array(p['subjects'])[keep][ev], adjusted=adjusted, unadjusted=original))
    save(RUN/'motion'/f'{model}.json', dict(status='complete', rows=rows, splits=splits,
        n=int(keep.sum()), motion_only=rs.identification_scores(-abs(fa[:, None]-fb[None, :]))))


def random_ready(model, co):
    return model == 'brainlm13' or all(complete(random_folder(model, s, co)/'complete.json') for s in SEEDS)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, rows = [], []
    for co in COHORTS:
        tasks.append(dict(id='summaries/'+co, status='complete' if complete(RUN/'summaries'/co/'complete.json') else 'pending'))
    for model in MODELS:
        for seed in SEEDS:
            for co in COHORTS:
                ok = model == 'brainlm13' or complete(random_folder(model, seed, co)/'complete.json')
                tasks.append(dict(id=f'random/{model}/{seed}/{co}', status='reused' if model == 'brainlm13' else 'complete' if ok else 'pending'))
        for co in COHORTS:
            for panel in ('signal', 'random'):
                path = RUN/'evaluation'/panel/model/f'{co}.json'
                ok = complete(path)
                tasks.append(dict(id=f'{panel}/{model}/{co}', status='complete' if ok else 'pending'))
                if ok: rows.extend(read(path)['rows'])
        path = RUN/'motion'/f'{model}.json'
        ok = complete(path)
        tasks.append(dict(id='motion/'+model, status='complete' if ok else 'pending'))
        if ok: rows.extend(read(path)['rows'])
    panel_done = {panel: all(t['status'] == 'complete' for t in tasks if t['id'].startswith(panel+'/')) for panel in ('signal', 'random', 'motion')}
    # Random extraction rows also use random/ prefix; reused13M rows count as done.
    panel_done['random'] = all(t['status'] in ('complete', 'reused') for t in tasks if t['id'].startswith('random/'))
    df = pd.DataFrame(rows)
    if len(df):
        df['holm_p'] = np.nan
        df['inference_status'] = 'pending_complete_panel'
        for family, ix in df.groupby('family').groups.items():
            if panel_done[family.split('::')[0]]:
                df.loc[ix, 'holm_p'] = holm(df.loc[ix, 'permutation_p'])
                df.loc[ix, 'inference_status'] = 'final_nine_model_control_family'
        df.to_csv(OUT/'Comparisons.csv', index=False)
    status = dict(updated=utc(), complete=sum(t['status'] in ('complete', 'reused') for t in tasks),
        total=len(tasks), panel_complete=panel_done, tasks=tasks, automatic_shutdown=False)
    save(RUN/'queue_status.json', status)
    if all(panel_done.values()):
        with pd.ExcelWriter(OUT/'Nine_model_identity_controls.xlsx', engine='xlsxwriter') as w:
            df.to_excel(w, sheet_name='Paired comparisons', index=False)
            pd.DataFrame(tasks).to_excel(w, sheet_name='Coverage', index=False)
            pd.DataFrame([dict(item=k, value=str(v)) for k, v in read(RUN/'protocol.json').items()]).to_excel(w, sheet_name='Protocol', index=False)
    return status


def cpu_job(kind, model, co):
    path = (RUN/'summaries'/co/'complete.json' if kind == 'summaries' else
        RUN/'motion'/f'{model}.json' if kind == 'motion' else RUN/'evaluation'/kind/model/f'{co}.json')
    if complete(path): return
    if kind == 'summaries': summaries(co)
    elif kind == 'motion': motion(model)
    else: evaluate(model, co, kind)


def cpu_worker(workers):
    lock = (RUN/'cpu.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    active = {}
    failed = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            jobs = []
            for co in COHORTS:
                if not complete(RUN/'summaries'/co/'complete.json'): jobs.append(('summaries', '', co))
            for model in MODELS:
                for co in COHORTS:
                    if complete(RUN/'summaries'/co/'complete.json') and not complete(RUN/'evaluation/signal'/model/f'{co}.json'):
                        jobs.append(('signal', model, co))
                    if random_ready(model, co) and not complete(RUN/'evaluation/random'/model/f'{co}.json'):
                        jobs.append(('random', model, co))
                if random_ready(model, 'psopt') and not complete(RUN/'motion'/f'{model}.json'):
                    jobs.append(('motion', model, 'psopt'))
            for fut in list(active):
                if fut.done():
                    job = active.pop(fut)
                    try: fut.result()
                    except Exception:
                        failed.add(job)
                        save(RUN/'errors'/('_'.join(job)+'.json'), dict(error=traceback.format_exc(), updated=utc()))
                        print('FAILED', job, traceback.format_exc(), flush=True)
            for job in jobs:
                if len(active) >= workers: break
                if job not in active.values() and job not in failed:
                    active[pool.submit(cpu_job, *job)] = job
            status = report()
            save(RUN/'cpu_status.json', dict(status='running', pid=os.getpid(), active=list(active.values()), failed=list(failed), updated=utc()))
            if status['complete'] == status['total']: break
            gpu = RUN/'gpu_status.json'
            if not active and (failed or (gpu.exists() and read(gpu)['status'] in ('complete', 'failed'))): break
            time.sleep(10)
    save(RUN/'cpu_status.json', dict(status='failed' if failed else 'complete' if report()['complete'] == report()['total'] else 'waiting_gpu', updated=utc()))


def gpu_worker(workers, device):
    lock = (RUN/'gpu.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable in this execution environment'
    # Each model runs in an isolated process to avoid upstream module-name clashes.
    order = ['brainlm111', 'brainlm650', 'fmrilm_mae', 'fmrilm_vq_domain', 'fmrilm', 'brainjepa', 'semantoks', 'brainharmonix_f']
    jobs = [('smoke', m, None) for m in order]
    jobs += [('extract', m, seed) for seed in SEEDS for m in order]
    active, failed = [], []
    LOG.mkdir(parents=True, exist_ok=True)
    while jobs or active:
        for entry in active.copy():
            proc, job, log = entry
            if proc.poll() is not None:
                active.remove(entry); log.close()
                if proc.returncode:
                    failed.append(job)
                    jobs = [j for j in jobs if j[1] != job[1]]
        while jobs and len(active) < workers:
            job = jobs.pop(0)
            action, model, seed = job
            if action == 'smoke' and complete(RUN/'smoke'/f'{model}.json'): continue
            if action == 'extract' and all(complete(random_folder(model, seed, co)/'complete.json') for co in COHORTS): continue
            # Never bypass parity checks while another worker is running the smoke.
            if action == 'extract' and not complete(RUN/'smoke'/f'{model}.json'):
                jobs.append(job)
                break
            log = (LOG/f'{action}_{model}_{seed}.log').open('a')
            cmd = [sys.executable, '-u', __file__, action, '--model', model, '--device', device]
            if seed is not None: cmd += ['--seed', str(seed)]
            active.append((subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), job, log))
        save(RUN/'gpu_status.json', dict(status='running', pid=os.getpid(), active=[dict(pid=p.pid, job=j) for p, j, _ in active], pending_jobs=len(jobs), failed=failed, updated=utc()))
        time.sleep(5)
    save(RUN/'gpu_status.json', dict(status='failed' if failed else 'complete', failed=failed, updated=utc()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['register', 'smoke', 'extract', 'cpu', 'gpu', 'report'])
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--seed', type=int, choices=SEEDS)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if a.action == 'register': register(); report()
    elif a.action == 'smoke': smoke(a.model, a.device)
    elif a.action == 'extract': random_extract(a.model, a.seed, a.device)
    elif a.action == 'cpu': cpu_worker(a.workers)
    elif a.action == 'gpu': gpu_worker(a.workers, a.device)
    else: print(json.dumps(report(), indent=2))


if __name__ == '__main__': main()
