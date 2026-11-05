#!/usr/bin/env python3
"""Download pinned public weights with size/SHA256 checks; never replace old models."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/new_fm_v1'
ASSETS = {
    'fmrilm': ['vq-contrastive/stage1-tokenizer.pt', 'vq-contrastive/stage2-pretrain-Qwen3-0.6B.pt'],
    'semantoks': ['brainsemantoks_ckpt_epoch_100.pth', 'config_used.yaml', 'network_mapping.npz',
                  'schaefer400_tian50_buckner7_MNI152_2mm.nii.gz'],
}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def fetch(job):
    name, meta, item = job
    file = item['rfilename']
    dest = ROOT / 'checkpoints/pretrained/new_fm_v1' / name / file
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('lfs', {}).get('sha256')
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + '.partial')
        url = f"https://huggingface.co/{meta['id']}/resolve/{meta['sha']}/{file}"
        with requests.get(url, stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            with tmp.open('wb') as f:
                for block in r.iter_content(8 * 1024 * 1024): f.write(block)
        if tmp.stat().st_size != item['size']: raise ValueError('Size mismatch: ' + file)
        if expected and digest(tmp) != expected: raise ValueError('Hash mismatch: ' + file)
        tmp.replace(dest)
    h = digest(dest)
    if dest.stat().st_size != item['size'] or (expected and h != expected):
        raise ValueError('Invalid existing checkpoint: ' + str(dest))
    print('verified', name, file, dest.stat().st_size, flush=True)
    return dict(path=str(dest), sha256=h, repository=meta['id'], revision=meta['sha'], file=file)

def main():
    RUN.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, names in ASSETS.items():
        meta = json.loads((ROOT / 'reproducibility/model_assets' / (name + '_hf_metadata.json')).read_text())
        items = {s['rfilename']: s for s in meta['siblings']}
        jobs.extend((name, meta, items[f]) for f in names)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: rows = list(pool.map(fetch, jobs))
    tmp = RUN / 'assets.tmp.json'
    tmp.write_text(json.dumps(dict(status='complete',files=rows), indent=2))
    tmp.replace(RUN / 'assets.json')

if __name__ == '__main__': main()
