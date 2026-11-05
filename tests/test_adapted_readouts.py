import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('adapted_readouts',ROOT/'scripts/run_adapted_readouts.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class AdaptedReadoutsTests(unittest.TestCase):
    def test_zero_adapter_requires_every_update_to_be_zero(self):
        d={'x.lora_A':torch.ones(2,4),'x.lora_B':torch.zeros(4,2)}
        self.assertTrue(m.zero_adapter(d));d['x.lora_B'][0,0]=1e-12;self.assertFalse(m.zero_adapter(d))

    def test_three_seed_ensemble_rejects_missing_or_misaligned_participants(self):
        rows=[dict(state=['adni','brainlm13','supervised','MEM',s],subjects=['a','b'],y=[0.,1.],predictions={'native':[i,i+1.]}) for i,s in enumerate(m.SEEDS)]
        np.testing.assert_allclose(m.aligned_ensemble(rows,'predictions')['native'],[1.,2.])
        with self.assertRaises(ValueError):m.aligned_ensemble(rows[:2],'predictions')
        rows[2]['subjects']=['b','a']
        with self.assertRaises(ValueError):m.aligned_ensemble(rows,'predictions')

    def test_matrix_ensemble_precedes_identification_scoring(self):
        from readout_sensitivity import identification_scores
        matrices=[[[1.,0.],[0.,1.]],[[1.,0.],[0.,1.]],[[0.,9.],[9.,0.]]]
        rows=[dict(state=['nacc','brainjepa','native_ssl','none',s],subjects=['a','b'],sim={'native':x}) for s,x in zip(m.SEEDS,matrices)]
        ensemble=m.aligned_ensemble(rows,'sim')['native']
        self.assertEqual(identification_scores(ensemble)['top1'],0.)
        self.assertAlmostEqual(np.mean([identification_scores(np.asarray(x))['top1'] for x in matrices]),2/3)

    def test_source_exposure_is_separated(self):
        s=('nacc','brainlm13','supervised','MEM',42)
        self.assertEqual(m.scope(s,'nacc'),'source_exposed_descriptive')
        self.assertEqual(m.scope(s,'nacc_repeated'),'source_exposed_descriptive')
        self.assertEqual(m.scope(s,'adni'),'external')

if __name__=='__main__':unittest.main()
