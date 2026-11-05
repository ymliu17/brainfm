import sys
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from fm_matched_v3 import supplement as s
from transfer_calibration import clinical_predict,subject_folds


class SupplementTests(unittest.TestCase):
    def test_recall_target_and_scan_timing_are_separate(self):
        m=pd.DataFrame(dict(MEM=[np.nan,np.nan],MEM_delayed_recall=[3.,7.]))
        key,y=s.outcome(m,'hrv_pre','MEM')
        self.assertEqual(key,'MEM_delayed_recall');np.testing.assert_array_equal(y,[3,7])
        self.assertTrue(m.MEM.isna().all())
        mask=s.baseline_rows(pd.DataFrame(dict(eligible=[True,True,False],visit=['T1','T2','T1'])),'act')
        np.testing.assert_array_equal(mask,[True,False,False])

    def test_loso_never_fits_to_held_cohort_outcomes(self):
        rng=np.random.default_rng(4);cohorts=s.COHORTS[2:]
        values={co:rng.normal(size=(25,8)) for co in cohorts}
        tables={co:pd.DataFrame(dict(subject=[f'{co}_{i}' for i in range(25)],visit='T1',eligible=True,
                                     MEM=rng.normal(size=25),EF=rng.normal(size=25))) for co in cohorts}
        def run(directory):
            with patch.object(s,'RUN',Path(directory)),patch.object(s,'representation',side_effect=lambda co,*a:values[co]),patch.object(s,'metadata',side_effect=lambda co:tables[co].copy()):
                s.loso('none','brainlm13','frozen','none',0)
                rows=json.loads((Path(directory)/'supplement/brainlm13_frozen/loso.json').read_text())['rows']
                return next(r for r in rows if r['cohort']=='hrv_pre' and r['domain']=='EF')
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            original=run(a);tables['hrv_pre']['EF']=rng.normal(100,500,size=25);changed=run(b)
        np.testing.assert_array_equal(original['prediction'],changed['prediction'])
        self.assertTrue(all(not x.startswith('hrv_pre:') for x in original['training_subjects']))

    def test_clinical_alignment_excludes_outer_test_statistics_and_labels(self):
        rng=np.random.default_rng(8);g=np.repeat(np.arange(30),2);n=len(g)
        C=rng.normal(size=(n,3));Z=rng.normal(size=(n,5));source=rng.normal(size=(40,5));y=rng.normal(size=n)
        tr,te=next(subject_folds(g,0))
        a,audit=clinical_predict(C,y,Z,source,g,'coral',0)
        altered=Z.copy();altered[te[1:]]=rng.normal(1000,100,size=(len(te)-1,5))
        yy=y.copy();yy[te]*=1000
        b,_=clinical_predict(C,yy,altered,source,g,'coral',0)
        self.assertAlmostEqual(a[te[0]],b[te[0]],places=10)
        for fold in audit:
            self.assertFalse(set(fold['calibration'])&set(fold['test']))
            self.assertFalse(set(fold['inner_calibration'])&set(fold['inner_validation']))


if __name__=='__main__':unittest.main()
