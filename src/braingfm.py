"""Frozen FC-based BrainGFM encoder with strict atlas and checkpoint validation."""
from pathlib import Path
import hashlib
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'checkpoints/pretrained/braingfm/f9739d729d1e/BrainGFM_pretrained.pth'
CHECKSUM = 'a4987a40779f579b7d1c56a262084ce8844ba33ddd2669d4c9a2c91b70edc355'
ATLASES = dict(schaefer=100, schaefer200=200, schaefer300=300, shen268=268,
               power264=264, gordon333=333, aal116=116, aal3v1=166)


def load_braingfm(checkpoint=CHECKPOINT, device='cpu', atlas='schaefer', seed=0):
    if atlas not in ATLASES:
        raise ValueError(f'Unsupported atlas {atlas}; choose {ATLASES}. A424 and Schaefer400+Tian450 are not supported.')
    checkpoint = Path(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != CHECKSUM:
        raise ValueError('Checkpoint differs from the audited snapshot; validate architecture before updating the adapter')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    enc = {k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')}
    if not enc or not all(torch.isfinite(v).all() for v in enc.values()):
        raise ValueError('Missing or non-finite encoder parameters')
    from vendor.braingfm import BrainGFM_Gprompt as upstream
    # All disease embeddings are checkpoint parameters. Avoid an unrelated BERT download
    # or silent random fallback before restoring their exact saved values.
    tokens = {k.removeprefix('disease_embedding_'): v.reshape(-1)
              for k, v in enc.items() if k.startswith('disease_embedding_')}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with patch.object(upstream, 'get_disease_embeddings', return_value={k: torch.zeros_like(v) for k,v in tokens.items()}):
            model = upstream.BrainGFM(ff_hidden_size=512, num_classes=2, num_self_att_layers=4,
                dropout=.2, num_GNN_layers=4, nhead=8, hidden_dim=256, max_feature_dim=512,
                rwse_steps=5, moe_num_experts=1, gcn_residual=True, gcn_norm=True,
                gcn_layer_norm=True, prenorm=True, attn_bias=True, readout='meanmax_ln',
                rwse_fixed=True, token_init=.02, node_id_emb=True)
        # Fail on ANY missing encoder parameter, including graph and node prompts.
        model.load_state_dict(enc, strict=True)
    model.eval().requires_grad_(False).to(device)
    model.braingfm_atlas = atlas
    model.benchmark_metadata = dict(family='BrainGFM', input='Pearson FC', atlas=atlas,
        disease_token='none', readout='meanmax_ln', dimensions=512, adjacency='FC > 0.3',
        checkpoint_sha256=digest, encoder_tensors=len(enc), strict_load=True,
        source_commit='f9739d729d1e3d186486ef8056337867c5fcfcf3',
        pretraining_overlap='not established; ADNI scores must not be claimed contamination-free')
    return model


def validate_fc(fc, atlas):
    a = np.asarray(fc, dtype=np.float32)
    n = ATLASES.get(atlas)
    if n is None or a.ndim != 3 or a.shape[1:] != (n,n):
        raise ValueError(f'Expected [subjects,{n},{n}] Pearson FC for {atlas}; got {a.shape}')
    if not np.isfinite(a).all() or np.max(np.abs(a)) > 1.00001:
        raise ValueError('Input must be finite Pearson correlation, not Fisher-z FC')
    if not np.allclose(a, a.transpose(0,2,1), atol=1e-5):
        raise ValueError('FC must be symmetric')
    if not np.allclose(np.diagonal(a, axis1=1, axis2=2), 1, atol=1e-5):
        raise ValueError('Pearson FC diagonal must be one; check constant or missing ROIs')
    return a


@torch.inference_mode()
def braingfm_fc_readout(model, fc, batch=4):
    fc = validate_fc(fc, model.braingfm_atlas)
    if batch < 1 or len(fc) == 0:
        raise ValueError('Nonempty input and positive batch size required')
    device = next(model.parameters()).device
    result = []
    for start in range(0,len(fc),batch):
        x = torch.as_tensor(fc[start:start+batch], device=device)
        result.append(model(x, (x > .3).float(), parc_type=model.braingfm_atlas,
                            disease_type='none').cpu().numpy())
    out = np.concatenate(result)
    if out.shape != (len(fc),512) or not np.isfinite(out).all():
        raise ValueError('Invalid BrainGFM embedding')
    return out


def pearson_fc(timeseries, valid=None):
    x = np.asarray(timeseries)
    if x.ndim != 3:
        raise ValueError('Expected [subjects,ROI,time]')
    v = np.full(len(x),x.shape[-1]) if valid is None else np.asarray(valid)
    if len(v) != len(x) or np.any(v < 3) or np.any(v > x.shape[-1]):
        raise ValueError('Invalid scan lengths')
    out = []
    for a, length in zip(x,v):
        a = np.asarray(a[:,:int(length)],float)
        if not np.isfinite(a).all() or np.any(a.std(axis=1) < 1e-10):
            raise ValueError('Non-finite scan or constant ROI')
        out.append(np.corrcoef(a).astype(np.float32))
    return np.stack(out)


def fc_edges(fc, atlas):
    a = validate_fc(fc,atlas)
    return np.arctanh(np.clip(a[:,np.triu_indices(a.shape[1],1)[0],np.triu_indices(a.shape[1],1)[1]],-.999999,.999999))
