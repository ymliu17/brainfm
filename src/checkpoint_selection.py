"""Resolve selection metadata for canonical exports and legacy checkpoints."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def resolve_selection(checkpoint,source,target,seed=42):
    checkpoint=Path(checkpoint);meta_path=checkpoint.with_suffix('.meta.json')
    meta=load(meta_path) if meta_path.exists() else {}
    if 'canonical_result' in meta:
        expected=dict(source=source,target=target,seed=seed,objective='supervised')
        if meta.get('status')!='complete' or any(meta.get(k)!=v for k,v in expected.items()):
            raise ValueError('Canonical export metadata does not match requested state')
        if sha(checkpoint)!=meta['sha256']:
            raise ValueError('Exported checkpoint hash mismatch')
        adapter=Path(meta['canonical_checkpoint']);result_path=Path(meta['canonical_result'])
        if sha(adapter)!=meta['canonical_checkpoint_sha256']:
            raise ValueError('Canonical adapter hash mismatch')
        if sha(result_path.parents[4]/'intake.json')!=meta['canonical_intake_sha256']:
            raise ValueError('Canonical intake hash mismatch')
        result=load(result_path)
        if result.get('status')!='complete' or len(result.get('folds',[]))!=1:
            raise ValueError('Expected one completed external-export fold')
        fold=result['folds'][0]
        expected=dict(source=source,model='brainlm13',target=target,seed=seed,mode='external')
        if fold.get('status')!='complete' or any(fold.get(k)!=v for k,v in expected.items()):
            raise ValueError('Canonical selection fold does not match exported encoder')
        selection=dict(status='complete',**expected,best_epoch=fold['best_epoch'],
            inner_val_frozen=fold['validation_frozen'],inner_val_selected=fold['validation_selected'],
            independent_test=False,metadata_origin='canonical_matrix',
            canonical_result=str(result_path),canonical_result_sha256=sha(result_path),
            canonical_checkpoint=str(adapter),canonical_checkpoint_sha256=sha(adapter),
            export_metadata_sha256=sha(meta_path),exported_checkpoint_sha256=sha(checkpoint))
    else:
        # Older standalone runs legitimately keep their result next to encoder.pt.
        # A broken canonical sidecar must never silently fall back to this path.
        if any(k.startswith('canonical_') for k in meta):
            raise ValueError('Incomplete canonical export metadata')
        result_path=checkpoint.parent/'result.json';selection=load(result_path)
        expected=dict(target=target,seed=seed,mode='external')
        if any(selection.get(k)!=v for k,v in expected.items()):
            raise ValueError('Legacy selection result does not match requested state')
        selection=dict(selection,metadata_origin='legacy_standalone',
            result_path=str(result_path),result_sha256=sha(result_path),
            exported_checkpoint_sha256=sha(checkpoint))
    epoch=selection.get('best_epoch')
    if not isinstance(epoch,int) or isinstance(epoch,bool) or epoch<0:
        raise ValueError('Invalid selected epoch')
    return selection


def selection_note(selection):
    if selection is None:return ''
    note=f"{selection.get('source','nacc').upper()} {selection['target']} seed{selection['seed']} selected epoch{selection['best_epoch']} using source-only validation."
    if selection['best_epoch']==0:note+=' Epoch zero is not evidence of an encoder update.'
    return note
