"""Binary diagnosis extension to the immutable four-model cognitive matrix.

Uses the existing common-atlas scans; adds versioned labels without changing the
matrix intake or invalidating cognitive checkpoints. No ordinal regression here.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from fm_matrix.data import ROOT, WORK, DATA, RUN as MATRIX_RUN, dataset, save, sha
from fm_matrix.models import Encoder, MODELS
from fm_matrix.train import Projection, amp, features

RUN = ROOT / 'runs/diagnosis_posttraining_v1'
SPECS = (('adni', 'DX'), ('nacc', 'DX'), ('nacc', 'AD_DX'))
SEEDS = (42, 43, 44)
DEFINITIONS = {
    'adni_DX': 'Scan-visit CN=0 versus Dementia=1; existing ADNI DX definition, not a separately verified etiologic label.',
    'nacc_DX': 'NACCUDSD 1=0 versus 4=1 (all-cause dementia); MCI and impaired-not-MCI excluded.',
    'nacc_AD_DX': 'CN=0 versus AD-etiology dementia=1; same clinical month and concordant NACCUDSD/NORMCOG/DEMENTED; dementia additionally NACCALZD=1.',
}


def make_splits(m, target, seed, mode):
    ids = np.flatnonzero(m[target].notna())
    y = m[target].to_numpy(float); g = m.subject.astype(str).to_numpy()
    if set(y[ids]) != {0., 1.}:
        raise ValueError('Diagnosis requires both binary classes')
    outer = [(np.arange(len(ids)), np.array([], int))] if mode == 'external' else list(
        StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(ids, y[ids], g[ids]))
    result = []
    for k, (pool, test) in enumerate(outer):
        pool = ids[pool]; test = ids[test]
        a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed*100+k).split(pool, y[pool], g[pool]))
        train, val = pool[a], pool[b]
        sets = [set(g[ix]) for ix in (train, val, test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError('Participant leakage')
        for ix in (train, val, test):
            if len(ix) and set(y[ix]) != {0., 1.}:
                raise ValueError('Single-class diagnosis split')
        result.append(dict(train=train.tolist(), validation=val.tolist(), test=test.tolist()))
    return result


def prepare():
    if (RUN/'protocol.json').exists():
        verify(); return
    RUN.mkdir(parents=True, exist_ok=True)
    inputs = {}; counts = {}
    for source in ('adni', 'nacc'):
        p = DATA/f'{source}.csv'; inputs[str(p)] = sha(p)
        m = pd.read_csv(p)[['subject', 'scan']].copy()
        targets = ['DX'] if source == 'adni' else ['DX', 'AD_DX']
        for target in targets:
            if source == 'adni':
                p = WORK/'data/derived/adni424_t200/ADNI_manifest_raw_DX_all_t200_off.csv'
                d = pd.read_csv(p); key = 'FILE_NAME'
            else:
                folder = 'nacc_parity_v1' if target == 'DX' else 'nacc_uds_v1'
                p = WORK/f'data/derived/{folder}/ADNI_manifest_nacc_{target}.csv'
                d = pd.read_csv(p); d['scan'] = d.subject.astype(str)+'_'+d.session.astype(str); key = 'scan'
            inputs[str(p)] = sha(p)
            if d[key].duplicated().any(): raise ValueError('Ambiguous scan labels')
            m[target] = m.scan.map(d.set_index(key).label)
            selected = m[m[target].notna()]
            counts[f'{source}_{target}'] = dict(scans=len(selected), subjects=selected.subject.nunique(), classes=selected[target].value_counts().to_dict(), definition=DEFINITIONS[f'{source}_{target}'])
            for seed in SEEDS:
                for mode in ('external', 'cv'):
                    save(RUN/'splits'/f'{source}_{target}_s{seed}_{mode}.json', make_splits(m, target, seed, mode))
        m.to_csv(RUN/f'{source}_labels.csv', index=False)
    files = [Path(__file__), ROOT/'scripts/run_diagnosis_posttraining.py', ROOT/'scripts/materialize_diagnosis_encoder.py', ROOT/'src/fm_matrix/models.py', ROOT/'src/fm_matrix/train.py', ROOT/'src/fm_matrix/data.py', ROOT/'src/model_input_protocol.py', ROOT/'src/roi_signal_qc.py']
    save(RUN/'protocol.json', dict(protocol='matched_binary_diagnosis_v1', status='preliminary', matrix_intake_sha256=sha(MATRIX_RUN/'intake.json'), counts=counts, input_sha256=inputs,
        label_sha256={s: sha(RUN/f'{s}_labels.csv') for s in ('adni', 'nacc')},
        split_sha256={p.name: sha(p) for p in sorted((RUN/'splits').glob('*.json'))}, code_sha256={str(p):sha(p) for p in files},
        models=list(MODELS), seeds=SEEDS, rank=8, alpha=16, effective_batch=8, microbatch=1, lr=1e-4, max_epochs=15, patience=5,
        head='Training-only StandardScaler/PCA<=100 and fixed C=1 logistic initialization; class-balanced sample weights; weighted BCEWithLogits during adaptation.',
        selection='Inner-validation ROC AUC, epoch zero eligible; five participant-isolated outer folds, same splits for all models.',
        transfer='External source-fitted encoders AND heads applied without target-label fitting. ADNI DX <-> NACC DX primary label mapping; NACC AD_DX -> ADNI DX is a distinct-case-definition sensitivity.',
        automatic_shutdown=False))


def verify():
    p = json.loads((RUN/'protocol.json').read_text())
    if sha(MATRIX_RUN/'intake.json') != p['matrix_intake_sha256']: raise ValueError('Matrix intake changed')
    for path, expected in {**p['input_sha256'], **p['code_sha256']}.items():
        if sha(path) != expected: raise ValueError('Registered diagnosis input/code changed: '+path)
    for s, expected in p['label_sha256'].items():
        if sha(RUN/f'{s}_labels.csv') != expected: raise ValueError('Diagnosis labels changed')
    for f, expected in p['split_sha256'].items():
        if sha(RUN/'splits'/f) != expected: raise ValueError('Diagnosis split changed')
    return p


def labels(source):
    m = pd.read_csv(RUN/f'{source}_labels.csv')
    original = pd.read_csv(DATA/f'{source}.csv')
    if not m[['subject','scan']].equals(original[['subject','scan']]): raise ValueError('Label/input order mismatch')
    return m


def location(source, model, target, seed, mode):
    return RUN/'training'/source/model/f'{target}_s{seed}_{mode}'


def metrics(y, p):
    return dict(auc=float(roc_auc_score(y,p)), average_precision=float(average_precision_score(y,p)),
                balanced_accuracy=float(balanced_accuracy_score(y,np.asarray(p)>=.5)), brier=float(brier_score_loss(y,p)))


def train(source, model, target, seed, mode, device='cuda', epochs=15):
    verify()
    root = location(source,model,target,seed,mode)
    if (root/'complete.json').exists(): return
    x, _ = dataset(source,model); m = labels(source); y=m[target].to_numpy(float); g=m.subject.astype(str).to_numpy()
    split_path=RUN/'splits'/f'{source}_{target}_s{seed}_{mode}.json'
    folds=json.loads(split_path.read_text()); rows=[]
    # No target outcomes are used in training or checkpoint selection.
    ext_source = 'nacc' if source=='adni' else 'adni'
    ext_target = 'DX'
    ex=em=None
    if mode=='external':
        ex,_=dataset(ext_source,model); em=labels(ext_source); ei=np.flatnonzero(em[ext_target].notna())
    for fold,split in enumerate(folds):
        out=root/f'fold{fold}'
        if (out/'result.json').exists(): rows.append(json.loads((out/'result.json').read_text())); continue
        tr,va,te=[np.asarray(split[k],int) for k in ('train','validation','test')]
        torch.manual_seed(seed*100+fold); rng=np.random.default_rng(seed*100+fold)
        net=Encoder(model,device); ft=features(net,x[tr],device); project=Projection(ft).to(device)
        z=project(torch.as_tensor(ft,device=device)).detach().cpu().numpy()
        # Mean balanced weights keep loss scale similar across cohorts/folds.
        class_weights=len(tr)/(2*np.bincount(y[tr].astype(int),minlength=2))
        clf=LogisticRegression(C=1.,class_weight='balanced',max_iter=5000).fit(z,y[tr])
        head=nn.Linear(project.dim,1).to(device)
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(clf.coef_,device=device)); head.bias.copy_(torch.as_tensor(clf.intercept_,device=device))
        frozen_head=copy.deepcopy(head.state_dict()); frozen_adapter=net.adapter_state()
        @torch.no_grad()
        def predict(tensor,ids):
            f=features(net,tensor[ids],device)
            return torch.sigmoid(head(project(torch.as_tensor(f,device=device))).squeeze(1)).cpu().numpy()
        p0=predict(x,te) if len(te) else []
        e0=predict(ex,ei) if mode=='external' else []
        v0=roc_auc_score(y[va],predict(x,va)); best=v0; best_epoch=0; stale=0; history=[]
        adapter=frozen_adapter; headstate=copy.deepcopy(frozen_head)
        params=[p for p in net.parameters() if p.requires_grad]+list(head.parameters())
        opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        for epoch in range(1,epochs+1):
            losses=[]; order=rng.permutation(tr)
            for start in range(0,len(order),8):
                ids=order[start:start+8]; opt.zero_grad()
                for idx in ids:
                    with amp(device):
                        logit=head(project(net(x[idx:idx+1].to(device)))).squeeze()
                        loss=nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor(y[idx],dtype=logit.dtype,device=device))*float(class_weights[int(y[idx])])
                    if not torch.isfinite(loss): raise ValueError('Nonfinite classification loss')
                    (loss/len(ids)).backward(); losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
            val=float(roc_auc_score(y[va],predict(x,va)))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_auc=val)); save(out/'history.json',history)
            print(source,model,target,seed,mode,fold,epoch,'validation_auc',val,flush=True)
            if val>best:
                best=val; best_epoch=epoch; stale=0; adapter=net.adapter_state(); headstate=copy.deepcopy(head.state_dict())
            else: stale+=1
            if stale>=5: break
        net.load_adapter(adapter); head.load_state_dict(headstate)
        pa=predict(x,te) if len(te) else []; ea=predict(ex,ei) if mode=='external' else []
        out.mkdir(parents=True,exist_ok=True)
        torch.save(dict(adapter=adapter,model=model,provenance=net.provenance(),intake_sha256=sha(MATRIX_RUN/'intake.json'),diagnosis_protocol_sha256=sha(RUN/'protocol.json'),source_checkpoint_sha256=sha(net.checkpoint_path),best_epoch=best_epoch),out/'adapter.pt')
        torch.save(dict(projection=project.cpu().state_dict(),head={k:v.cpu() for k,v in headstate.items()},frozen_head={k:v.cpu() for k,v in frozen_head.items()},output='sigmoid probability'),out/'head.pt')
        result=dict(status='complete',source=source,model=model,target=target,seed=seed,mode=mode,fold=fold,train_subjects=g[tr],validation_subjects=g[va],test_subjects=g[te],test_indices=te,y=y[te],frozen_prediction=p0,adapted_prediction=pa,best_epoch=best_epoch,validation_frozen_auc=v0,validation_selected_auc=best,history=history,class_weights=class_weights,split_sha256=sha(split_path))
        if len(te): result.update(frozen=metrics(y[te],p0),adapted=metrics(y[te],pa))
        if mode=='external':
            ey=em[ext_target].to_numpy(float)[ei]
            result['external']=dict(cohort=ext_source,target=ext_target,definition=DEFINITIONS[f'{ext_source}_{ext_target}'],case_definition_sensitivity=target=='AD_DX',subjects=em.subject.to_numpy()[ei],scans=em.scan.to_numpy()[ei],y=ey,frozen_prediction=e0,adapted_prediction=ea,frozen=metrics(ey,e0),adapted=metrics(ey,ea))
        save(out/'result.json',result); rows.append(result)
        del net,project,head,opt,params; torch.cuda.empty_cache()
    summary=dict(status='complete',folds=rows,protocol_sha256=sha(RUN/'protocol.json'),max_epochs=epochs,scope='Source-only validation plus external fixed-head transfer' if mode=='external' else 'Independent fold-specific encoder training; pooled session-weighted OOF metrics')
    if mode=='cv':
        yy=np.concatenate([r['y'] for r in rows]); f=np.concatenate([r['frozen_prediction'] for r in rows]); a=np.concatenate([r['adapted_prediction'] for r in rows])
        summary.update(frozen=metrics(yy,f),adapted=metrics(yy,a),delta_auc=float(roc_auc_score(yy,a)-roc_auc_score(yy,f)))
    save(root/'complete.json',summary)


def delegate_legacy(args):
    """Retire duplicate 13M fitting while preserving standalone encoder consumers."""
    if args.target not in ('DX','AD_DX') or not (RUN/'protocol.json').exists(): return False
    if not args.out_root.resolve().is_relative_to(WORK/'runs'): return False
    if (args.epochs,args.patience,args.batch_size)!=(15,5,8): raise ValueError('Legacy diagnosis budget differs')
    old=pd.read_csv(args.data_dir/f'ADNI_manifest_nacc_{args.target}.csv')
    old['scan']=old.subject.astype(str)+'_'+old.session.astype(str)
    current=labels('nacc').dropna(subset=[args.target])
    a=old.set_index('scan').label.sort_index(); b=current.set_index('scan')[args.target].sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.to_numpy(),b.to_numpy()):
        raise ValueError('Legacy diagnosis population differs; cannot silently delegate')
    train('nacc','brainlm13',args.target,args.seed,args.mode)
    out=args.out_root/f'{args.target}_{args.mode}_seed{args.seed}'
    canonical=location('nacc','brainlm13',args.target,args.seed,args.mode)
    if args.mode=='external':
        from materialize_diagnosis_encoder import materialize
        materialize(args.target,args.seed,out/'encoder.pt')
    save(out/'delegation.json',dict(status='complete',operation='Delegated to canonical four-model diagnosis protocol; no duplicate fitting',canonical_result=str(canonical/'complete.json'),legacy_recipe_retired=True))
    return True
