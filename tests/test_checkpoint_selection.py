import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from checkpoint_selection import resolve_selection,selection_note,sha


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.ck=self.root/'export/encoder.pt';self.ck.parent.mkdir();self.ck.write_bytes(b'encoder')
        self.result=self.root/'matrix/training/nacc/brainlm13/supervised_MEM_s42_external/complete.json'
        self.result.parent.mkdir(parents=True);self.adapter=self.result.parent/'fold0/adapter.pt'
        self.adapter.parent.mkdir();self.adapter.write_bytes(b'adapter')
        self.intake=self.root/'matrix/intake.json';self.intake.write_text('{}')
        self.fold=dict(status='complete',source='nacc',model='brainlm13',target='MEM',seed=42,mode='external',best_epoch=3,validation_frozen=.2,validation_selected=.3)
        self.result.write_text(json.dumps(dict(status='complete',folds=[self.fold])))
        self.meta=dict(status='complete',source='nacc',target='MEM',objective='supervised',seed=42,sha256=sha(self.ck),canonical_result=str(self.result),canonical_checkpoint=str(self.adapter),canonical_checkpoint_sha256=sha(self.adapter),canonical_intake_sha256=sha(self.intake))
        self.meta_path=self.ck.with_suffix('.meta.json');self.meta_path.write_text(json.dumps(self.meta))

    def tearDown(self):self.temp.cleanup()

    def test_canonical_selection_without_legacy_result(self):
        s=resolve_selection(self.ck,'nacc','MEM');self.assertEqual(s['best_epoch'],3)
        self.assertEqual(s['metadata_origin'],'canonical_matrix')
        self.assertIn('epoch3',selection_note(s));self.assertNotIn('epoch0',selection_note(s))

    def test_canonical_metadata_takes_precedence_over_stale_legacy_result(self):
        (self.ck.parent/'result.json').write_text(json.dumps(dict(target='MEM',seed=42,mode='external',best_epoch=0)))
        self.assertEqual(resolve_selection(self.ck,'nacc','MEM')['best_epoch'],3)

    def test_corrupt_checkpoint_adapter_or_intake_rejected(self):
        for p in [self.ck,self.adapter,self.intake]:
            before=p.read_bytes();p.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):resolve_selection(self.ck,'nacc','MEM')
            p.write_bytes(before)

    def test_missing_or_mismatched_canonical_result_never_falls_back(self):
        (self.ck.parent/'result.json').write_text(json.dumps(dict(target='MEM',seed=42,mode='external',best_epoch=0)))
        self.fold['target']='EF';self.result.write_text(json.dumps(dict(status='complete',folds=[self.fold])))
        with self.assertRaises(ValueError):resolve_selection(self.ck,'nacc','MEM')
        self.result.unlink()
        with self.assertRaises(FileNotFoundError):resolve_selection(self.ck,'nacc','MEM')

    def test_legacy_exports_still_work_and_epoch_zero_is_conditional(self):
        self.meta_path.unlink();(self.ck.parent/'result.json').write_text(json.dumps(dict(target='MEM',seed=42,mode='external',best_epoch=0)))
        s=resolve_selection(self.ck,'nacc','MEM');self.assertEqual(s['metadata_origin'],'legacy_standalone')
        self.assertIn('not evidence of an encoder update',selection_note(s))
        self.assertEqual(selection_note(None),'')

    @unittest.skipUnless(all((ROOT/f'runs/unified_input_v3/workspace/runs/nacc_parity_v1/posttraining/{target}_external_seed42/encoder.pt').exists() for target in ['MEM','EF']), 'Requires private experiment artifacts; not distributed')
    def test_real_mem_and_ef_exports_resolve_to_actual_completed_folds(self):
        for target in ['MEM','EF']:
            ck=ROOT/f'runs/unified_input_v3/workspace/runs/nacc_parity_v1/posttraining/{target}_external_seed42/encoder.pt'
            s=resolve_selection(ck,'nacc',target)
            fold=json.loads(Path(s['canonical_result']).read_text())['folds'][0]
            self.assertEqual(s['best_epoch'],fold['best_epoch'])
            self.assertEqual(s['inner_val_selected'],fold['validation_selected'])


if __name__=='__main__':unittest.main()
