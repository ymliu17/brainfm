#!/usr/bin/env python3
"""Explicitly fetch the recorded external source revisions, without weights."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = json.loads((ROOT / 'reproducibility/upstream_sources.json').read_text())
    for item in records:
        dest = ROOT / item['destination']
        if dest.exists():
            raise SystemExit(f'Refusing to overwrite {dest}; inspect the existing checkout first.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', item['repository'], str(dest)], check=True)
        subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', item['revision']], check=True)
        actual = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != item['revision']:
            raise RuntimeError(f'Unexpected revision in {dest}')
        print(f'{item["destination"]}: {actual}')


if __name__ == '__main__':
    main()
