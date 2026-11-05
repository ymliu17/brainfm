"""Audit newly supplied FD without changing registered benchmark inputs."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from motion_covariates import attach_fd, sha

OUT=ROOT/'runs/motion_covariates_v1'
COHORTS=('adni','nacc','nacc_repeated','act','cogte','breathe','psopt','hrv_pre','hrv_post')

def main():
    (OUT/'labels').mkdir(parents=True,exist_ok=True)
    sources={};rows=[];missing=[];pairs=[];extremes=[];tables={}
    for co in COHORTS:
        path=ROOT/'data/derived/fm_matched_v3'/f'{co}.csv'
        m,audit=attach_fd(pd.read_csv(path),co)
        tables[co]=m
        audit.update(canonical_manifest=str(path.relative_to(ROOT)),canonical_sha256=sha(path))
        dest=OUT/'labels'/f'{co}.csv';m.to_csv(dest,index=False)
        audit['output_sha256']=sha(dest);sources[co]=audit
        missing.extend(m.loc[~m.fd_available,['cohort','subject','scan','visit','fd_missing_reason']].to_dict('records'))
        extremes.extend(m.loc[m.mean_fd>1,['cohort','subject','scan','visit','mean_fd']].to_dict('records'))
        pairs_path=ROOT/'runs/brainlm13_identity_controls_v1/pairs'/f'{co}.json'
        pair_n=pair_fd=None
        if pairs_path.exists():
            p=json.loads(pairs_path.read_text());idx=m.set_index('scan');pair_n=len(p['subjects'])
            a=idx.loc[p['a']['scans'],'mean_fd'].to_numpy();b=idx.loc[p['b']['scans'],'mean_fd'].to_numpy()
            valid=np.isfinite(a)&np.isfinite(b);pair_fd=int(valid.sum())
            for s,sa,sb,fa,fb,ok in zip(p['subjects'],p['a']['scans'],p['b']['scans'],a,b,valid):
                pairs.append(dict(cohort=co,subject=s,scan_a=sa,scan_b=sb,mean_fd_a=fa,mean_fd_b=fb,paired_fd_available=bool(ok)))
            sources[co]['pair_manifest_sha256']=sha(pairs_path)
        rows.append(dict(cohort=co,scans=len(m),subjects=m.subject.nunique(),matched_fd_scans=int(m.fd_available.sum()),
                         missing_fd_scans=int((~m.fd_available).sum()),reid_pairs=pair_n,paired_fd_available=pair_fd,
                         mean_fd_min=m.mean_fd.min(),mean_fd_median=m.mean_fd.median(),mean_fd_max=m.mean_fd.max(),
                         fd_gt_1_descriptive_count=int((m.mean_fd>1).sum()),
                         frame_count_mismatches=int(m.fd_frame_count_difference.dropna().ne(0).sum()) if 'fd_frame_count_difference' in m else None))
    pairs_path=ROOT/'runs/brainlm13_identity_controls_v1/pairs/hrv.json'
    p=json.loads(pairs_path.read_text())
    a=tables[p['a']['cohort']].set_index('scan').loc[p['a']['scans'],'mean_fd'].to_numpy()
    b=tables[p['b']['cohort']].set_index('scan').loc[p['b']['scans'],'mean_fd'].to_numpy()
    valid=np.isfinite(a)&np.isfinite(b)
    for s,sa,sb,fa,fb,ok in zip(p['subjects'],p['a']['scans'],p['b']['scans'],a,b,valid):
        pairs.append(dict(cohort='hrv',subject=s,scan_a=sa,scan_b=sb,mean_fd_a=fa,mean_fd_b=fb,paired_fd_available=bool(ok)))
    row=next(r for r in rows if r['cohort']=='hrv_post')
    row.update(reid_pairs=len(p['subjects']),paired_fd_available=int(valid.sum()))
    sources['hrv_post']['pair_manifest_sha256']=sha(pairs_path)
    pd.DataFrame(rows).to_csv(OUT/'Coverage.csv',index=False)
    pd.DataFrame(missing).to_csv(OUT/'Missing_scans.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'ReID_pairs.csv',index=False)
    pd.DataFrame(extremes).to_csv(OUT/'FD_over_1_descriptive.csv',index=False)
    protocol=dict(status='intake_complete',sources=sources,
        primary_eligibility_changed=False,experiments_rerun=False,
        exclusions='None. Missing FD restricts only an explicitly requested matched sensitivity panel. No FD threshold or motion_outliers exclusion.',
        resolution='Supplied scan-average FD only; per-frame FD is not available for matching model-truncated input windows.',
        units='Source FD values retained without conversion. No new framewise displacement computation.',
        duplicates='Collapse exact duplicate source rows only. Conflicting scan-key records fail.',
        coverage='NACC selected and repeated manifests overlap; their counts must not be summed. Re-ID pair eligibility is audited separately.',
        hrv_er='Uploaded subject-level pre/post FD matched separately to existing older-adult scans; Re-ID coverage is reported on the hrv_post row. Additional source participants are not added to the benchmark.',
        analysis_use='Use attach_fd for additive analyses. Fit residualization within training or separate calibration participants; compare adjusted/unadjusted methods on identical FD-complete samples. Registered historical loaders and results remain unchanged.',
        automatic_shutdown=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (OUT/'README.txt').write_text('FD intake for supplementary motion analyses.\nSee Coverage.csv, Missing_scans.csv, ReID_pairs.csv and protocol.json.\nNo primary sample exclusions, model fitting, new test results, or manuscript changes.\nADNI/NACC frame counts are checked; other sources do not supply frame counts.\nFD_over_1_descriptive.csv flags values for inspection only; it is not an exclusion list.\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
