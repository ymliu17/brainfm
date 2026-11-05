import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts')]
from run_sf100_loso import fit_held,COHORTS,populations


class LOSOTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(27)
        self.arrays={co:rng.normal(size=(20,8)) for co in COHORTS}
        self.tables={co:pd.DataFrame(dict(subject=[str(i) for i in range(20)],scan=[co+str(i) for i in range(20)],MEM=rng.normal(size=20),EF=rng.normal(size=20))) for co in COHORTS}
        self.masks={co:np.ones(20,bool) for co in COHORTS}

    def test_held_out_outcomes_do_not_fit_prediction(self):
        a=fit_held(self.arrays,self.tables,self.masks,'EF','cogte')
        self.tables['cogte']['EF']=np.arange(20)[::-1]*100.
        b=fit_held(self.arrays,self.tables,self.masks,'EF','cogte')
        np.testing.assert_array_equal(a['prediction'],b['prediction'])
        self.assertFalse(np.array_equal(a['y'],b['y']))
        self.assertFalse(any(s.startswith('cogte:') for s in a['training_subjects']))

    def test_unavailable_scan_excluded_from_training_and_testing(self):
        self.masks['cogte'][0]=False;self.arrays['cogte'][0]=np.nan
        a=fit_held(self.arrays,self.tables,self.masks,'MEM','act')
        self.assertNotIn('cogte:0',a['training_subjects'])
        b=fit_held(self.arrays,self.tables,self.masks,'MEM','cogte')
        self.assertNotIn('cogte0',b['scan_ids']);self.assertEqual(b['n'],19)
        self.assertTrue(np.isfinite(b['prediction']).all())

    def test_hrv_recall_is_not_added_to_baseline_memory(self):
        cohorts,_=populations(self.tables,self.masks,'MEM','act')
        self.assertNotIn('hrv_pre',cohorts)
        with self.assertRaises(ValueError):populations(self.tables,self.masks,'MEM','hrv_pre')

if __name__=='__main__':unittest.main()
