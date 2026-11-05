#!/usr/bin/env python3
"""Execute one isolated fMRI-LM variant task using the matched parent engines."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['mae', 'vq-domain'], required=True)
    parser.add_argument('action', choices=['validate', 'features', 'supervised', 'native_ssl', 'local', 'axes', 'loso', 'identification', 'paired_visit', 'readouts', 'canonical', 'harmonization', 'short_features', 'fusion', 'robustness', 'report'])
    parser.add_argument('--source', default='none')
    parser.add_argument('--model', default='fmrilm')
    parser.add_argument('--objective', default='frozen')
    parser.add_argument('--target', default='none')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', default='external')
    parser.add_argument('--cohort', default='adni')
    parser.add_argument('--phase', default='frozen')
    parser.add_argument('--device', default='cpu')
    a = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    from fmrilm_variants.runtime import configure
    run, Encoder = configure(a.variant)
    state = (a.source, 'fmrilm', a.objective, a.target, a.seed)
    if a.action == 'validate':
        from new_fm.data import dataset, save
        x, _ = dataset('adni', 'fmrilm')
        rows = []
        for ssl in (False, True):
            net = Encoder(device=a.device, native_ssl=ssl)
            batch = x[:2].to(a.device)
            with torch.no_grad():
                first = net(batch)
                second = net(batch)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            assert first.shape == (2, 1024) and torch.isfinite(first).all()
            loss = None
            if ssl:
                l1 = net.ssl_loss(batch, 123)
                l2 = net.ssl_loss(batch, 123)
                torch.testing.assert_close(l1, l2, rtol=0, atol=0)
                assert torch.isfinite(l1)
                l1.backward()
                gradients = [p.grad for n, p in net.named_parameters() if 'lora_B' in n and p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert any(g.abs().max() > 0 for g in gradients)
                loss = float(l1.detach())
            rows.append(dict(ssl=ssl, feature_shape=list(first.shape), loss=loss, encoder=net.provenance()))
            del net
            if a.device == 'cuda':
                torch.cuda.empty_cache()
        save(run / 'compatibility.json', dict(status='complete', variant=a.variant, checks=rows,
             strict_weights=True, deterministic_inference=True, deterministic_validation_loss=True,
             finite_adapter_gradients=True))
    elif a.action in ('supervised', 'native_ssl'):
        from new_fm import train
        if a.action == 'supervised':
            train.supervised(a.source, 'fmrilm', a.target, a.seed, a.mode, a.device)
        else:
            train.ssl(a.source, 'fmrilm', a.seed, a.device)
    elif a.action == 'features':
        from new_fm.extract import extract
        extract(state, a.device)
    elif a.action in ('short_features', 'fusion', 'robustness'):
        from new_fm import sensitivity
        getattr(sensitivity, a.action)('fmrilm', a.cohort, a.device)
    elif a.action == 'report':
        from new_fm.report import report
        report(a.phase)
    else:
        from new_fm import evaluate
        if a.action == 'local':
            if a.cohort == 'all':
                from new_fm.data import COHORTS, save
                for co in COHORTS[:-1]:
                    evaluate.local(state, co)
                save(evaluate.rowdir(state) / 'local_complete.json', dict(status='complete'))
            else:
                evaluate.local(state, a.cohort)
        elif a.action == 'readouts':
            evaluate.readouts('fmrilm')
        else:
            getattr(evaluate, a.action)(state)


if __name__ == '__main__':
    main()
