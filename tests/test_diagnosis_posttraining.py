import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import diagnosis_posttraining as d
from run_diagnosis_posttraining import tasks


class DiagnosisTests(unittest.TestCase):
    @unittest.skipUnless((d.RUN/'protocol.json').exists(), 'Requires private experiment artifacts; not distributed')
    def test_actual_splits_disjoint_balanced_complete_and_identical(self):
        d.verify()
        for source,target in d.SPECS:
            m=d.labels(source)
            for seed in d.SEEDS:
                for mode in ('cv','external'):
                    folds=d.make_splits(m,target,seed,mode)
                    self.assertEqual(folds,json.loads((d.RUN/'splits'/f'{source}_{target}_s{seed}_{mode}.json').read_text()))
                    seen=[]
                    for f in folds:
                        groups=[set(m.iloc[f[k]].subject) for k in ('train','validation','test')]
                        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
                        for k,ix in f.items():
                            if ix:self.assertEqual(set(m.iloc[ix][target]),{0.,1.})
                        seen+=f['test']
                    if mode=='cv':self.assertEqual(sorted(seen),np.flatnonzero(m[target].notna()).tolist())

    def test_equal_model_coverage_and_no_ordinal_task(self):
        jobs=tasks();self.assertEqual(len(jobs),72)
        signatures=[]
        for model in d.MODELS:
            signatures.append({(j['source'],j['target'],j['seed'],j['mode']) for j in jobs if j['model']==model})
        self.assertTrue(all(s==signatures[0] for s in signatures))
        self.assertEqual(sum(5 if j['mode']=='cv' else 1 for j in jobs),216)

    def test_classifier_training_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'protocol.json').write_text('{}');(root/'intake.json').write_text('{}');(root/'base.pt').write_bytes(b'base')
            torch.manual_seed(17);x=torch.randn(60,8)
            m=pd.DataFrame(dict(subject=[str(i) for i in range(60)],scan=[str(i) for i in range(60)],DX=np.tile([0.,1.],30)))
            d.save(root/'splits/nacc_DX_s42_cv.json',d.make_splits(m,'DX',42,'cv'))
            class TinyEncoder(nn.Module):
                def __init__(self,model,device):
                    super().__init__();self.linear=nn.Linear(8,6);self.checkpoint_path=root/'base.pt'
                def forward(self,x):return self.linear(x)
                def adapter_state(self):return {k:v.detach().clone() for k,v in self.state_dict().items()}
                def load_adapter(self,state):self.load_state_dict(state)
                def provenance(self):return {'test_double':True}
            with patch.object(d,'RUN',root),patch.object(d,'MATRIX_RUN',root),patch.object(d,'verify'),patch.object(d,'dataset',return_value=(x,m)),patch.object(d,'labels',return_value=m),patch.object(d,'Encoder',TinyEncoder):
                d.train('nacc','brainlm13','DX',42,'cv',device='cpu',epochs=1)
                p=d.location('nacc','brainlm13','DX',42,'cv')/'complete.json';result=json.loads(p.read_text())
                self.assertEqual(len(result['folds']),5)
                self.assertTrue(np.isfinite(result['delta_auc']))
                for f in result['folds']:
                    self.assertGreater(len(f['history']),0)
                    self.assertTrue(np.isfinite(f['history'][0]['loss']))
                    self.assertTrue(all(0<=v<=1 for v in f['adapted_prediction']))
                before=p.stat().st_mtime_ns
                d.train('nacc','brainlm13','DX',42,'cv',device='cpu',epochs=1)
                self.assertEqual(before,p.stat().st_mtime_ns)

    def test_single_class_rejected_and_auc_not_correlation(self):
        m=pd.DataFrame(dict(subject=list('abcdef'),DX=[0.]*6))
        with self.assertRaises(ValueError):d.make_splits(m,'DX',42,'cv')
        a=d.metrics([0,0,1,1],[.1,.2,.8,.9]);self.assertEqual(a['auc'],1.)

    @unittest.skipUnless((ROOT/'runs/unified_input_v3/queue_status.json').exists(), 'Requires private experiment artifacts; not distributed')
    def test_actual_legacy_jobs_delegate_without_second_recipe(self):
        status=json.loads((ROOT/'runs/unified_input_v3/queue_status.json').read_text())
        selected=[]
        for job in status['tasks']:
            cmd=job['command']
            if Path(cmd[1]).name!='nacc_posttrain.py' or '--target' not in cmd:continue
            value=lambda k,default:cmd[cmd.index('--'+k)+1] if '--'+k in cmd else default
            target=value('target',None)
            if target not in ('DX','AD_DX'):continue
            a=SimpleNamespace(target=target,mode=value('mode',None),seed=int(value('seed',42)),epochs=15,patience=5,batch_size=8,
                data_dir=Path(value('data-dir',d.WORK/'data/derived/nacc_parity_v1')),
                out_root=Path(value('out-root',d.WORK/'runs/nacc_parity_v1/posttraining')))
            with patch.object(d,'train') as fit,patch.object(d,'save'),patch('materialize_diagnosis_encoder.materialize') as merge:
                self.assertTrue(d.delegate_legacy(a));fit.assert_called_once_with('nacc','brainlm13',target,a.seed,a.mode)
                self.assertEqual(merge.call_count,int(a.mode=='external'))
            selected.append(job['name'])
        self.assertEqual(len(selected),12)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
