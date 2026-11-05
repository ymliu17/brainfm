"""Reuse immutable audited inputs, with separately validated HRV domain labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fm_matrix.data import ROOT, WORK, save, sha, clean
from fm_matrix import data as original

RUN = ROOT/'runs/fm_matched_v3'
DATA = ROOT/'data/derived/fm_matched_v3'
COMPOSITES = DATA/'hrv_domain_composites.csv'
COMPOSITE_METADATA = COMPOSITES.with_suffix('.json')
MEMORY = DATA/'hrv_delayed_recall.csv'


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    if (RUN/'intake.json').exists():
        return
    old = json.loads((original.RUN/'intake.json').read_text())
    counts = {}
    for name, count in old['counts'].items():
        source = original.DATA/f'{name}.csv'
        if sha(source) != count['manifest_sha256']:
            raise ValueError('Original manifest changed')
        (DATA/source.name).write_bytes(source.read_bytes())
        for atlas in ('a424', 'sch450'):
            dest = DATA/f'{name}_{atlas}.pt'
            if not dest.exists():
                dest.symlink_to(original.DATA/dest.name)
        counts[name] = dict(count)
    save(RUN/'intake.json', dict(status='complete', counts=counts,
         parent_intake=str(original.RUN/'intake.json'), parent_sha256=sha(original.RUN/'intake.json'),
         inputs='Unchanged canonical tensors; new models use a common readout',
         hrv_composites='Pending validated domain-composite file; named-test v2 outputs are not relabeled'))


def dataset(name, model):
    # Original loader verifies both tensor and original-manifest checksums.
    x, _ = original.dataset(name, model)
    m = pd.read_csv(DATA/f'{name}.csv')
    expected = json.loads((RUN/'intake.json').read_text())['counts'][name]['manifest_sha256']
    if sha(DATA/f'{name}.csv') != expected:
        raise ValueError('New manifest changed without registration')
    return x, m


def endpoints(name):
    return ['EF'] if name.startswith('hrv') else ['MEM', 'EF']


def hrv_ef_scores(table, visit):
    """Study-defined mean of three uncorrected standard scores; require all three."""
    columns = [f'{visit}_{test}_standardScore' for test in ('flanker', 'LSWM', 'PCPS')]
    values = table[columns].apply(pd.to_numeric, errors='raise')
    if np.isinf(values.to_numpy()).any():
        raise ValueError('Infinite NIH Toolbox score')
    return values.mean(axis=1, skipna=False)


def build_hrv_composites():
    import zipfile
    source = ROOT/'data/HRV_biofeedback/label/phenotype.zip'
    with zipfile.ZipFile(source) as z:
        table = pd.read_csv(z.open('NIHcognition_summary.tsv'), sep='\t')
    rows = []
    for visit in ('pre', 'post'):
        rows.append(pd.DataFrame(dict(subject=table.participant_id, visit=visit,
                                      MEM=np.nan, EF=hrv_ef_scores(table, visit))))
    DATA.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(COMPOSITES, index=False)
    meta = dict(source=str(source), source_sha256=sha(source),
        standardized_using='Published uncorrected standardScore columns; no sample-fitted standardization',
        available_endpoints=['EF'],
        definitions=dict(EF='Study-defined arithmetic mean of Flanker, List Sorting Working Memory, and Pattern Comparison Processing Speed uncorrected standard scores. All three required, at each visit.',
                         MEM='Unavailable: no independent episodic-memory composite in NIHcognition_summary. Working memory is included in EF, not relabeled as MEM.'),
        missing_policy='Any missing component makes EF missing. No imputation.',
        picture_memory='Picture-memory tasks are separate from NIH scores and from pre/post MEM composites.')
    if MEMORY.exists():
        audit=json.loads((RUN/'hrv_memory.json').read_text())
        if audit['status']!='complete' or sha(MEMORY)!=audit['score_sha256']:
            raise ValueError('Unverified delayed-recall scores')
        meta['available_endpoints'].append('MEM_delayed_recall')
        meta['definitions']['MEM_delayed_recall']=audit['definition']
        meta['memory_score_sha256']=sha(MEMORY)
        meta['memory_timing']=audit['source_timing']
        meta['memory_analysis']=audit['analysis']
    save(COMPOSITE_METADATA,meta)
    install_hrv_composites()


def hrv_labels_ready():
    if not (RUN/'hrv_composites.json').exists():
        return False
    stamp = json.loads((RUN/'hrv_composites.json').read_text())
    memory_ok = (not stamp.get('memory_file_sha256') or
                 (MEMORY.exists() and sha(MEMORY)==stamp['memory_file_sha256']))
    return (memory_ok and stamp.get('status') == 'complete' and COMPOSITES.exists() and COMPOSITE_METADATA.exists()
            and sha(COMPOSITES) == stamp['file_sha256']
            and sha(COMPOSITE_METADATA) == stamp['metadata_sha256'])


def install_hrv_composites():
    """Install documented EF labels; retain unavailable MEM explicitly as missing."""
    if not COMPOSITES.exists() or not COMPOSITE_METADATA.exists():
        raise ValueError('HRV EF source scores and documented construction are required')
    meta = json.loads(COMPOSITE_METADATA.read_text())
    for key in ['source', 'standardized_using']:
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise ValueError('Missing composite metadata: '+key)
    for target in ['MEM', 'EF']:
        if not isinstance(meta.get('definitions', {}).get(target), str) or not meta['definitions'][target].strip():
            raise ValueError('Missing scientifically defined composite: '+target)
    labels = pd.read_csv(COMPOSITES)
    if not {'subject', 'visit', 'MEM', 'EF'} <= set(labels):
        raise ValueError('Expected subject, visit, MEM and EF')
    if labels[['subject', 'visit']].isna().any().any() or labels.duplicated(['subject', 'visit']).any():
        raise ValueError('Invalid or duplicate participant/visit')
    if not set(labels.visit) <= {'pre', 'post'}:
        raise ValueError('Visits must be pre/post')
    for target in ['MEM', 'EF']:
        labels[target] = pd.to_numeric(labels[target], errors='raise')
        if np.isinf(labels[target]).any():
            raise ValueError('Infinite composite score')
    if labels.MEM.notna().any() or meta.get('available_endpoints') not in (['EF'],['EF','MEM_delayed_recall']):
        raise ValueError('MEM composite/pre-post scores unavailable; delayed recall is a separate endpoint')
    memory=None
    if 'MEM_delayed_recall' in meta['available_endpoints']:
        if sha(MEMORY)!=meta['memory_score_sha256']:
            raise ValueError('Delayed-recall score file changed')
        memory=pd.read_csv(MEMORY).set_index('subject')
        if memory.index.has_duplicates or not memory.MEM_delayed_recall.between(0,36).all():
            raise ValueError('Invalid delayed-recall labels')
    labels = labels.set_index(['subject', 'visit'])
    staged = {}
    for name, visit in [('hrv_pre', 'pre'), ('hrv_post', 'post')]:
        m = pd.read_csv(original.DATA/f'{name}.csv')
        for target in ['MEM', 'EF']:
            for tp, prefix in [('pre', 'baseline_'), ('post', 'follow_')]:
                index = pd.MultiIndex.from_arrays([m.subject, [tp]*len(m)])
                m[prefix+target] = labels[target].reindex(index).to_numpy()
            m[target] = m[('baseline_' if visit == 'pre' else 'follow_')+target]
        if not m.EF.notna().any():
            raise ValueError('No usable domain labels for '+name)
        if memory is not None:
            m['MEM_delayed_recall']=m.subject.map(memory.MEM_delayed_recall)
            m['MEM_delayed_recall_week']=5
            m['MRI_study_week']=2 if visit=='pre' else 7
        staged[name] = m
    for name, m in staged.items():
        m.to_csv(DATA/f'{name}_domains.csv', index=False)
    save(RUN/'hrv_composites.json', dict(status='complete', file_sha256=sha(COMPOSITES),
         metadata_sha256=sha(COMPOSITE_METADATA), definitions=meta,
         memory_file_sha256=sha(MEMORY) if memory is not None else None,
         manifests={n: sha(DATA/f'{n}_domains.csv') for n in staged},
         counts={n: {t: int(m[t].notna().sum()) for t in ['MEM', 'EF','MEM_delayed_recall'] if t in m} for n, m in staged.items()}))


def metadata(name):
    if name.startswith('hrv'):
        if not hrv_labels_ready():
            raise ValueError('HRV domain composites are pending; do not relabel individual tests')
        path = DATA/f'{name}_domains.csv'
        stamp = json.loads((RUN/'hrv_composites.json').read_text())
        if sha(path) != stamp['manifests'][name]:
            raise ValueError('HRV composite manifest changed')
        return pd.read_csv(path)
    return pd.read_csv(DATA/f'{name}.csv')


def splits(source, target, seed, mode):
    # Supervised sources and label masks are unchanged from the canonical pool.
    if source not in ('adni', 'nacc'):
        raise ValueError('Only observational source cohorts can train encoders')
    return original.splits(source, target, seed, mode)
