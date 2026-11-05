import sys,unittest
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_nacc_uds import enrich
class UdsJoinTests(unittest.TestCase):
 def frames(self):
  m=pd.DataFrame({'subject':['sub-A','sub-B','sub-C'],'age':[70,80,75]})
  c=pd.DataFrame({'ID':['sub-A','sub-B','sub-C'],'VISITYR':[2024]*3,'VISITMO':[5]*3,'NACCUDSD':[1,4,4]})
  u=pd.DataFrame({'NACCID':['C','B','A'],'VISITYR':[2010,2024,2024],'VISITMO':[5]*3,'NACCAGE':[61,80,70],'SEX':[1,2,1],'EDUC':[99,18,16],'DEMENTED':[1,1,0],'NORMCOG':[0,0,1],'NACCALZD':[1,1,8]})
  return m,c,u
 def test_visit_guard_and_row_order(self):
  m,c,u=self.frames();z=enrich(m,c,u)
  self.assertEqual(z.subject.tolist(),m.subject.tolist());self.assertEqual(z.age.tolist(),[70,80,75]);self.assertEqual(z.AD_DX.iloc[:2].tolist(),[0,1]);self.assertTrue(np.isnan(z.AD_DX.iloc[2]))
 def test_unknown_education_and_sex_mapping(self):
  z=enrich(*self.frames());self.assertEqual(z.SEX_binary_male.tolist(),[1,0,1]);self.assertTrue(np.isnan(z.education_years.iloc[2]))
 def test_duplicate_participants_rejected(self):
  m,c,u=self.frames()
  with self.assertRaises(ValueError):enrich(m,c,pd.concat([u,u.iloc[:1]]))
if __name__=='__main__':unittest.main()
