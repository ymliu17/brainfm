import sys
from pathlib import Path
import unittest
import pytest
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from motion_covariates import validate_unique, attach_fd

class MotionIntakeTests(unittest.TestCase):
    def test_duplicates_require_agreement(self):
        d=pd.DataFrame({'scan':['a','a'],'mean_fd':[.2,.2]})
        self.assertEqual(len(validate_unique(d)),1)
        d.loc[1,'mean_fd']=.3
        with self.assertRaises(ValueError):validate_unique(d)

    def test_invalid_motion_rejected(self):
        for val in [-.1,float('inf')]:
            with self.assertRaises(ValueError):validate_unique(pd.DataFrame({'scan':['a'],'mean_fd':[val]}))

    @pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/"data/HRV_biofeedback/HRV_per_subject_pre_post_mean_FD.csv").exists(), reason="Requires restricted cohort FD files")
    def test_actual_visit_and_run_mapping(self):
        m=pd.DataFrame({'scan':['sub-0105_T1','sub-0105_T2']})
        result,_=attach_fd(m,'act')
        self.assertEqual(result.source_session.tolist(),['ses-01','ses-03'])
        self.assertNotIn('mean_fd',m)
        result,_=attach_fd(pd.DataFrame({'scan':['sub-101_T1']}),'breathe')
        self.assertEqual(result.source_fd_column.iloc[0],'FD_rest_run1')

    @pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/"data/HRV_biofeedback/HRV_per_subject_pre_post_mean_FD.csv").exists(), reason="Requires restricted cohort FD files")
    def test_missing_subject_not_borrowed(self):
        result,_=attach_fd(pd.DataFrame({'scan':['sub-1022_T1','sub-1001_T2']}),'cogte')
        self.assertFalse(result.fd_available.iloc[0])
        self.assertEqual(result.source_session.iloc[1],'ses-02')

    @pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/"data/HRV_biofeedback/HRV_per_subject_pre_post_mean_FD.csv").exists(), reason="Requires restricted cohort FD files")
    def test_hrv_visits_are_distinct(self):
        root=Path(__file__).resolve().parents[1]
        source=pd.read_csv(root/'data/HRV_biofeedback/HRV_per_subject_pre_post_mean_FD.csv').set_index('subject')
        m=pd.DataFrame({'scan':['sub-7003_pre','sub-7003_post']})
        result,_=attach_fd(m,'hrv_pre')
        self.assertEqual(result.mean_fd.tolist(),source.loc['sub-7003',['mean_fd_pre_mm','mean_fd_post_mm']].tolist())
        subject=source.index[source.mean_fd_post_mm.isna()][0]
        result,_=attach_fd(pd.DataFrame({'scan':[subject+'_post']}),'hrv_post')
        self.assertFalse(result.fd_available.iloc[0])
        self.assertEqual(result.fd_missing_reason.iloc[0],'missing_source_value')

if __name__=='__main__':unittest.main()
