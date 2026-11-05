#!/usr/bin/env python3
"""Fetch both user-requested fMRI-LM variants at the original pinned revision."""
import concurrent.futures
import json
from pathlib import Path
from fetch_new_fm_assets import fetch

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta = json.loads((ROOT / 'reproducibility/model_assets/fmrilm_hf_metadata.json').read_text())
    wanted = {f'{variant}/{stage}' for variant in ('mae', 'vq-domain')
              for stage in ('stage1-tokenizer.pt', 'stage2-pretrain-Qwen3-0.6B.pt')}
    items = {r['rfilename']: r for r in meta['siblings']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(fetch, [('fmrilm', meta, items[name]) for name in sorted(wanted)]))
    dest = ROOT / 'runs/fmrilm_variants_v1/assets.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dict(status='complete', files=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
