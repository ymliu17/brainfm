"""Scope and numerical-equivalence checks for the bounded supplement."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import copy
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_adapted_readouts_focused as m

class FocusedTests(unittest.TestCase):
    def test_final_layer_operators_equal_original(self):
        z = torch.arange(2*5*3*7, dtype=torch.float32).reshape(2,5,3,7).sin()
        net = SimpleNamespace(tokens=lambda x: z)
        m.base.frozen.readout_names = m._original_names
        try:
            full = m._original_last(net, 'brainjepa', torch.zeros(2,1))
        finally:
            m.base.frozen.readout_names = lambda model: list(m.READOUTS)
        reduced = m.selected_last(net, 'brainjepa', torch.zeros(2,1))
        self.assertEqual(set(reduced), {'native', 'connectivity'})
        for k in m.READOUTS: np.testing.assert_array_equal(full[k], reduced[k])
        self.assertEqual(m.base.frozen.readout_names('brainjepa'), list(m.READOUTS))

    @unittest.skipUnless((m.OLD/'manifest.json').exists(), 'Requires private experiment artifacts; not distributed')
    def test_complete_uniform_scope_and_external_only_cognition(self):
        d = m.read(m.OLD/'manifest.json'); tasks = m.jobs(d)
        self.assertEqual(len(tasks), 648)
        self.assertEqual(sum(j['action']=='worker' for j in tasks),162)
        cognitive = [j for j in tasks if 'endpoint' in j]
        self.assertEqual(len(cognitive),108)
        self.assertTrue(all(j['cohort'] != j['state'][0] for j in cognitive))
        self.assertEqual(len({j['state'][1] for j in cognitive}),9)
        for r in d['states']:
            source=r['state'][0]
            self.assertTrue(all(p.parent.name!='cognition' for p in m.expected_outputs(r['state'],source)))

    @unittest.skipUnless(any((m.OLD/'inference').glob('*/*.json')), 'Requires private experiment artifacts; not distributed')
    def test_reuse_filter_preserves_predictions_statistics_and_identity_modes(self):
        files=sorted((m.OLD/'inference').glob('*/*.json'))
        for identity in (False,True):
            p=next(p for p in files if ('ensemble_predictions' not in m.read(p))==identity)
            original=m.read(p); reduced=m.relabel_inference(copy.deepcopy(original))
            self.assertTrue(reduced['rows'])
            for row in reduced['rows']:
                old=next(r for r in original['rows'] if all(r[k]==row[k] for k in ('candidate','reference','metric','mode')) and r['family'].split('::',1)[1]==row['family'].split('::',1)[1])
                for key in old:
                    if key!='family': self.assertEqual(row[key],old[key])
            if not identity:
                for key in m.READOUTS: np.testing.assert_array_equal(reduced['ensemble_predictions'][key],original['ensemble_predictions'][key])
            else:
                self.assertEqual({r['mode'] for r in reduced['scores']},{'native','projected128'})

    def test_source_exposure_stays_explicit_for_identification(self):
        s=('nacc','brainjepa','native_ssl','none',42)
        self.assertEqual(m.scope(s,'nacc_repeated'),'source_exposed_descriptive')
        self.assertEqual(m.scope(s,'adni'),'external')

if __name__=='__main__': unittest.main()
