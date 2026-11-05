"""Configure existing matched engines inside one isolated worker process."""
import json
import importlib
from pathlib import Path

from fm_matrix.data import save, sha
from fmrilm_variants.models import VariantEncoder

ROOT = Path(__file__).resolve().parents[2]


def configure(variant):
    if variant not in ('mae', 'vq-domain'):
        raise ValueError(variant)
    run = ROOT / 'runs/fmrilm_variants_v1' / variant
    run.mkdir(parents=True, exist_ok=True)
    parent = ROOT / 'runs/new_fm_v1/intake.json'
    d = json.loads(parent.read_text())
    d.update(checkpoint_variant=variant, parent_intake_sha256=sha(parent),
             scope_sha256=sha(ROOT / 'runs/fmrilm_variants_v1/scope.json'),
             evidence_status='variant-specific stage2 checkpoint; same disclosed scan-local robust input adapter')
    path = run / 'intake.json'
    if path.exists() and json.loads(path.read_text()) != d:
        raise ValueError('Variant intake changed')
    if not path.exists():
        save(path, d)
    class Encoder(VariantEncoder):
        pass
    Encoder.variant = variant
    for name in ('data', 'train', 'extract', 'evaluate', 'sensitivity', 'report'):
        module = importlib.import_module('new_fm.' + name)
        if hasattr(module, 'RUN'):
            module.RUN = run
        if hasattr(module, 'Encoder'):
            module.Encoder = Encoder
        if hasattr(module, 'MODELS'):
            module.MODELS = ('fmrilm',)
    return run, Encoder
