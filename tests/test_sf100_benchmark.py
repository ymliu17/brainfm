import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from sf100_benchmark import canonical_key,archive_key,choose_equivalent
import sf100_benchmark as benchmark
from paper_paired_stats import paired

class MatchingTests(unittest.TestCase):
    def test_scan_visit_alignment(self):
        self.assertEqual(canonical_key('adni',SimpleNamespace(subject='002_S_0295',visit='M060')),('ADNI','sub-ADNI002S0295','M060'))
        self.assertEqual(canonical_key('nacc',SimpleNamespace(subject='sub-NACC002039',visit=20240502)),archive_key(dict(dataset='NACC',subject='sub-NACC002039',visit='ses-20240502',session='ses-20240502')))
        self.assertEqual(canonical_key('hrv_post',SimpleNamespace(subject='sub-7003',visit='post')),('HRV-ER','sub-7003','T2'))

    def test_user_confirmed_act_followup_mapping(self):
        self.assertEqual(canonical_key('act',SimpleNamespace(subject='sub-0105',visit='T1'))[-1],'T1_baseline')
        self.assertEqual(canonical_key('act',SimpleNamespace(subject='sub-0105',visit='T2'))[-1],'T3_6mo')

    def test_distinct_runs_not_silently_collapsed(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.npy',Path(d)/'b.npy'
            np.save(a,np.eye(100));np.save(b,np.eye(100))
            self.assertEqual(choose_equivalent([b,a]),a)
            x=np.eye(100);x[0,1]=x[1,0]=.3;np.save(b,x)
            self.assertIsNone(choose_equivalent([a,b]))

    def test_repeated_participants_remain_outside_training(self):
        rng=np.random.default_rng(0);groups=np.repeat(np.arange(24),2);n=len(groups)
        m=pd.DataFrame(dict(subject=groups.astype(str),scan=['scan'+str(i) for i in range(n)],
                            eligible=True,age=60+groups,sex=(groups%2).astype(float)))
        x={name:rng.normal(size=(n,8)) for name in ('sf100_fc','braingfm')}
        def small_pair(*args,**kwargs):return paired(*args,**kwargs,n_boot=100,n_perm=100)
        with tempfile.TemporaryDirectory() as d,patch.object(benchmark,'RUN',Path(d)),patch.object(benchmark,'labels',return_value=m),patch.object(benchmark,'features',return_value=x),patch.object(benchmark,'matches',return_value=[{'available':True}]*n),patch.object(benchmark,'paired',side_effect=small_pair):
            benchmark.evaluate('adni')
            for name in ('AGE','SEX'):
                result=json.loads((Path(d)/'evaluation/adni'/f'{name}.json').read_text())
                self.assertEqual(result['status'],'complete')
                for split in result['split_audit']:
                    self.assertFalse(set(groups[split['train']])&set(groups[split['test']]))
                    self.assertFalse(set(groups[split['inner_train']])&set(groups[split['validation']]))
                for p in result['predictions'].values():self.assertTrue(np.isfinite(p).all())

if __name__=='__main__':unittest.main()
