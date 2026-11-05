#!/usr/bin/env python3
"""Retrieve pinned public picture-recall labels and derive one-time memory scores."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fm_matched_v3.data import DATA,RUN,save,sha

COMMIT='f88b2678e5944df96bb34aa19f967ce38b44b2f3'
RAW=ROOT/'data/HRV_biofeedback/label/openneuro_1.3.5_memory'


def recall_score(table,subject):
    """Unique correctly recalled pictures. Recorded zero recall is valid; absence is not."""
    if table.empty:
        raise ValueError('Empty file is not evidence of a completed zero-recall task')
    ids=table.participant_id.astype(str).str.replace('sub-','',regex=False)
    if not ids.eq(subject.removeprefix('sub-')).all():
        raise ValueError('Participant mismatch')
    correct=pd.to_numeric(table.correct,errors='raise')
    if correct.isna().any() or not set(correct)<=set([0,1,2]):
        raise ValueError('Unrecognized recall accuracy code')
    code=table.final_code.astype('string').str.strip()
    positive=correct.eq(1)
    if (positive & (code.isna()|code.str.lower().isin(['no','n/a','none','']))).any():
        raise ValueError('Correct response lacks a coded stimulus identity')
    n=code[positive].nunique()
    if n>36:raise ValueError('More correctly recalled pictures than encoded pictures')
    return dict(MEM_delayed_recall=int(n),n_responses=len(table),
                n_correct_rows=int(positive.sum()),duplicate_correct_responses=int(positive.sum()-n),
                n_incorrect_rows=int((~positive).sum()),encoded_set='|'.join(sorted(table['set'].dropna().astype(str).unique())))


def main():
    people=pd.read_csv(ROOT/'data/HRV_biofeedback/label/participants.tsv',sep='\t')
    people=people[people.age_group.eq('OA')].copy()
    tree=json.loads((RUN/'openneuro_1.3.5_tree.json').read_text())
    if tree.get('truncated') or tree['sha']!=COMMIT:
        raise ValueError('Wrong or incomplete pinned dataset tree')
    entries={r['path']:r for r in tree['tree']}
    RAW.mkdir(parents=True,exist_ok=True)
    selected=[];missing=[]
    for r in people.to_dict('records'):
        subject=r['participant_id'];key=f'{subject}/beh/{subject}_task-recall_beh.tsv'
        if key not in entries:
            missing.append(dict(subject=subject,declared_available=int(r['task-recall']),reason='No recall file in pinned public release'))
            continue
        selected.append((r,entries[key]))
    def fetch(pair):
        participant,entry=pair
        url='https://raw.githubusercontent.com/OpenNeuroDatasets/ds003823/'+COMMIT+'/'+entry['path']
        path=RAW/Path(entry['path']).name
        content=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=40).read()
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob!=entry['sha']:raise ValueError('Download checksum mismatch: '+entry['path'])
        if not path.exists():path.write_bytes(content)
        return participant,path,dict(path=str(path),url=url,git_blob=blob,sha256=sha(path))
    rows=[];files=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for participant,path,record in pool.map(fetch,selected):
            table=pd.read_csv(path,sep='\t');subject=participant['participant_id']
            result=recall_score(table,subject)
            if int(participant['task-recall'])!=1:
                raise ValueError('Task availability flag conflicts with file: '+subject)
            rows.append(dict(subject=subject,group=participant['condition'],**result));files.append(record)
    scores=pd.DataFrame(rows).sort_values('subject')
    DATA.mkdir(parents=True,exist_ok=True)
    output=DATA/'hrv_delayed_recall.csv';scores.to_csv(output,index=False)
    counts={}
    for cohort in ('hrv_pre','hrv_post'):
        m=pd.read_csv(DATA/(cohort+'.csv'))
        counts[cohort]=int(m.subject.isin(scores.subject).sum())
    save(RUN/'hrv_memory.json',dict(status='complete',dataset='ds003823',version='1.3.5',commit=COMMIT,
        files=files,score_file=str(output),score_sha256=sha(output),missing=missing,
        participants=len(scores),imaging_matches=counts,
        definition='Number of distinct correctly recalled pictures across all valences (0-36); higher is better. One-time delayed picture recall, not a MEM composite.',
        source_timing='Encoding at Week 4; delayed free recall at Week 5. Baseline MRI Week 2; post MRI Week 7.',
        timing_evidence=['https://www.nature.com/articles/s41597-023-02396-5',
                        'https://doi.org/10.1007/s10484-023-09579-1'],
        release_timing_note='Public recall TSVs have no week column; root recall dictionary describes matching to Week-4 images, and the release supplies one recall time point. Interpretation as Week-5 delayed recall follows these metadata and the published coding protocol.',
        accuracy_code_note='Dictionary lists 1=correct,2=incorrect; released files also use 0=incorrect. Only code 1 counts as correct; coded stimulus IDs are deduplicated.',
        missing_policy='Absent/empty/unscored files are not zero; explicit completed rows with no correct response yield zero.',
        analysis='Baseline MRI predicts later Week-5 recall; Week-7 MRI supports a separate retrospective association. No pre/post memory change or baseline-memory adjustment.'))
    print(json.dumps(dict(participants=len(scores),imaging_matches=counts,
                         range=[int(scores.MEM_delayed_recall.min()),int(scores.MEM_delayed_recall.max())],
                         mean=float(scores.MEM_delayed_recall.mean()),duplicates=int(scores.duplicate_correct_responses.sum()))))


if __name__=='__main__':main()
