#!/usr/bin/env python3
"""Append independent within-source fold selection metadata to technical audit."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import pandas as pd
import run_adapted_readouts as a
from fm_matrix.data import save,sha


def main():
    out=ROOT/'results/generated/readout_completion_v1';rows=[];histories=[];missing=[];files={}
    for s in a.states():
        source,model,obj,target,seed=s
        if obj!='supervised':continue
        root,internal=a.training_run(s)
        folder=root/'training'/source/internal/f'{obj}_{target}_s{seed}_cv'
        for fold in range(5):
            p=folder/f'fold{fold}'/'result.json'
            if not p.exists():missing.append(str(p));continue
            d=json.loads(p.read_text());files[str(p)]=sha(p)
            hp=p.parent/'history.json';h=json.loads(hp.read_text()) if hp.exists() else []
            if hp.exists():files[str(hp)]=sha(hp)
            rows.append(dict(model=model,source=source,objective=target,seed=seed,fold=fold,
                best_epoch=d['best_epoch'],initial_selected=d['best_epoch']==0,
                validation_before=d['validation_frozen'],validation_selected=d['validation_selected'],
                validation_improvement=d['validation_selected']-d['validation_frozen'],
                epochs_run=max([r['epoch'] for r in h],default=0),
                scope='Fold-specific participant-grouped within-source encoder evaluation; validation selects checkpoint, outer test excluded.'))
            histories.extend(dict(model=model,source=source,objective=target,seed=seed,fold=fold,**r) for r in h)
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['model','source','objective']).agg(checkpoints=('fold','size'),initial_selected=('initial_selected','sum'),
        median_best_epoch=('best_epoch','median'),mean_validation_improvement=('validation_improvement','mean')).reset_index()
    book=out/'Technical_coverage_and_checkpoint_audit.xlsx'
    tables={name:pd.read_csv(out/(name+'.csv')) for name in
            ('Checkpoint_selection','Selection_summary','Training_histories','Complete_context_coverage','Interpretation')}
    additions={'Within_source_CV_checkpoints':frame,'Within_source_CV_summary':summary,'CV_training_histories':pd.DataFrame(histories)}
    tables.update(additions)
    for name,d in additions.items():d.to_csv(out/(name+'.csv'),index=False)
    tmp=book.with_name(book.stem+'.tmp.xlsx')
    with pd.ExcelWriter(tmp,engine='xlsxwriter') as w:
        for name,d in tables.items():d.to_excel(w,sheet_name=name,index=False)
    tmp.replace(book)
    save(ROOT/'runs/readout_completion_v1/cv_audit_complete.json',dict(status='complete' if not missing else 'coverage_gap',
        expected=540,found=len(rows),initial_selected=int(frame.initial_selected.sum()),missing=missing,sources_sha256=files,
        script_sha256=sha(Path(__file__))))
    print(dict(expected=540,found=len(rows),missing=len(missing),initial_selected=int(frame.initial_selected.sum())))


if __name__=='__main__':main()
