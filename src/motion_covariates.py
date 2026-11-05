"""Scan-exact FD intake for additive sensitivity analyses.

Never modify canonical labels, primary eligibility, or historical results.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'adni': 'data/adni/adni_framewise_displacement_from_fmriprep.tsv',
    'nacc': 'data/NACC/nacc_framewise_displacement.csv',
    'act': 'data/rct/act/fd_all_ACT.csv',
    'cogte': 'data/rct/cogte/fd_T124_CogTE.csv',
    'breathe': 'data/rct/breathe/breathe_mean_fd_all_timepoints.csv',
    'psopt': 'data/rct/psopt/pSOPT_adsct_fc_meanFD.csv',
    'hrv': 'data/HRV_biofeedback/HRV_per_subject_pre_post_mean_FD.csv',
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_unique(table):
    """Collapse equivalent records only; reject conflicting FD or provenance."""
    table = table.drop_duplicates().copy()
    if table.scan.duplicated().any():
        raise ValueError('Ambiguous FD records: '+', '.join(table.loc[table.scan.duplicated(False),'scan']))
    values = pd.to_numeric(table.mean_fd, errors='raise')
    if (np.isinf(values) | (values < 0)).any():
        raise ValueError('FD must be finite and nonnegative, or explicitly missing')
    table['mean_fd'] = values
    return table

def normalize(cohort):
    co = 'nacc' if cohort == 'nacc_repeated' else 'hrv' if cohort in ('hrv_pre','hrv_post') else cohort
    path = ROOT/SOURCES[co]
    raw = pd.read_csv(path, sep='\t' if path.suffix=='.tsv' else ',')
    n_raw = len(raw)
    exact_duplicates = int(raw.duplicated().sum())
    d = raw.drop_duplicates().copy()
    if co=='adni':
        t = pd.DataFrame(dict(scan=d.subject+'_'+d.session, mean_fd=d.fd_mean,
                             source_session=d.session, source_fd_column='fd_mean',
                             source_n_frames=d.n_timepoints))
        mapping = 'Exact BIDS subject and session, including zero-padded Mxxx.'
    elif co=='nacc':
        t = pd.DataFrame(dict(scan='sub-'+d.ID+'_'+d.session.astype(str), mean_fd=d.mean_fd,
                             source_session=d.session.astype(str),source_fd_column='mean_fd',
                             source_n_frames=d.n_fd_values+1))
        if not d.n_confounds_files.eq(1).all():raise ValueError('NACC has pooled confounds files')
        mapping = 'Exact NACC ID and scan date; n_fd_values+1 used only for frame-count audit.'
    elif co=='act':
        visits = {'ses-01':'T1','ses-03':'T2'}
        d = d[d.Session.isin(visits)].copy()
        t = pd.DataFrame(dict(scan=d.ID+'_'+d.Session.map(visits),mean_fd=d.FD_mean,
                             source_session=d.Session,source_fd_column='FD_mean'))
        mapping = 'Canonical T1=ses-01; T2=ses-03 (previously user-confirmed T3_6mo follow-up). Later visits excluded.'
    elif co=='cogte':
        tables=[]
        for visit, session, fd, expected in [('T1','Session','FD_mean_01','ses-01'),('T2','Session_2','FD_mean_02','ses-02')]:
            if not d[session].eq(expected).all():raise ValueError('Unexpected CogTE session label')
            tables.append(pd.DataFrame(dict(scan=d.ID+'_'+visit,mean_fd=d[fd],
                                            source_session=d[session],source_fd_column=fd)))
        t = pd.concat(tables,ignore_index=True)
        mapping = 'T1=FD_mean_01/ses-01; T2=FD_mean_02/ses-02. Session 4 is not substituted.'
    elif co=='breathe':
        t = pd.DataFrame(dict(scan=d.ID+'_'+d.timepoint,mean_fd=d.FD_rest_run1,
                             source_session=d.timepoint,source_fd_column='FD_rest_run1'))
        mapping = 'Exact T1/T2 visit; resting-state run-01 per prior user confirmation. No run-02/task fallback.'
    elif co=='hrv':
        values=d[['mean_fd_pre_mm','mean_fd_post_mm']].notna()
        if not values.sum(axis=1).eq(d.n_scans).all() or not values.all(axis=1).astype(int).eq(d.has_both_scans).all():
            raise ValueError('HRV visit availability flags disagree with FD values')
        t=pd.concat([pd.DataFrame(dict(scan=d.subject+'_'+visit,mean_fd=d['mean_fd_'+visit+'_mm'],
                                      source_session=visit,source_fd_column='mean_fd_'+visit+'_mm'))
                     for visit in ('pre','post')],ignore_index=True)
        mapping='Exact subject plus pre/post; visit-specific mean FD in mm. Missing post is not filled from pre; availability flags validated, not used as exclusions.'
    else:
        t = pd.DataFrame(dict(scan='sub-'+d.subject.astype(str).str.removeprefix('sub-')+'_'+d.timepoint,
                             mean_fd=d.mean_fd,source_session=d.timepoint,source_fd_column='mean_fd'))
        mapping = 'Existing pSOPT mean_fd matched by subject and T1/T2 visit.'
    t = validate_unique(t)
    t['source_file'] = str(path.relative_to(ROOT))
    t['source_sha256'] = sha(path)
    return t, dict(source=str(path.relative_to(ROOT)),sha256=sha(path),source_rows=n_raw,
                   exact_duplicate_rows_removed=exact_duplicates,mapping=mapping)

def attach_fd(metadata, cohort):
    """Return copied scan metadata and source audit; no imputation or exclusion."""
    if metadata.scan.duplicated().any():raise ValueError('Canonical scan IDs are not unique')
    source, audit = normalize(cohort)
    index = source.set_index('scan')
    m = metadata.copy()
    if 'mean_fd' in m:raise ValueError('Refuse to overwrite an existing mean_fd column')
    m['fd_record_found'] = m.scan.isin(index.index)
    for col in source.columns.drop('scan'):m[col] = m.scan.map(index[col])
    m['fd_available'] = m.mean_fd.notna()
    m['fd_missing_reason'] = np.where(~m.fd_record_found,'no_matching_scan',np.where(~m.fd_available,'missing_source_value',''))
    if 'source_n_frames' in m:
        m['fd_frame_count_difference'] = m.source_n_frames-m.native_frames
    return m, audit
