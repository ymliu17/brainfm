"""Fresh-constructor controls for the nine frozen final-layer configurations.

Random arms retain fixed atlas/position priors, never checkpoint-learned tensors.
They remove all learned pretraining, including language pretraining in fMRI-LM.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class ControlEncoder(nn.Module):
    def __init__(self, model, seed=None):
        super().__init__()
        self.model = model
        self.seed = seed
        self.kind = 'fmrilm' if model.startswith('fmrilm') else model
        # Fork CPU RNG: a seed specifies the same encoder across every cohort/visit.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0 if seed is None else seed)
            self.build(seed is None)
        self.eval().requires_grad_(False)

    def build(self, pretrained):
        model = self.model
        if model == 'brainlm13':
            from run_identity_controls import create_encoder
            self.net, xyz = create_encoder(self.seed)
            self.register_buffer('xyz', xyz)
        elif model in ('brainlm111', 'brainlm650'):
            from paths import BRAINLM_111M, BRAINLM_650M, XYZ_PATH
            from brainlm_input import roi_order_by_y
            from vitmae_input import load_vitmae
            from transformers import ViTMAEConfig
            from brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
            folder = Path(BRAINLM_111M if model == 'brainlm111' else BRAINLM_650M).parent
            self.order = roi_order_by_y(XYZ_PATH)
            if pretrained:
                self.net, self.cfg = load_vitmae(folder)
            else:
                self.cfg = ViTMAEConfig.from_pretrained(folder)
                self.cfg.mask_ratio = 0.
                self.cfg.train_mode = 'inference'
                self.net = ViTMAEForPreTraining(self.cfg)
        elif model == 'brainjepa':
            from brainjepa import load_brainjepa, _official, _gradient, DEFAULT_GRAD
            from fm_matrix.models import efficient_jepa
            if pretrained:
                self.net = load_brainjepa(verbose=False)
            else:
                factory = _official()
                self.net, unused = factory(device='cpu', patch_size=16, crop_size=(450, 160),
                    pred_depth=12, pred_emb_dim=384, model_name='vit_base',
                    gradient_pos_embed=_gradient(DEFAULT_GRAD, 'cpu'), attn_mode='normal',
                    add_w='mapping', gradient_checkpointing=False)
                del unused
                g = self.net.gradient_pos_embed
                del self.net.gradient_pos_embed
                self.net.register_buffer('gradient_pos_embed', g, persistent=False)
            efficient_jepa(self.net)
        elif model == 'semantoks':
            from new_fm.models import semantoks_backbone, ASSETS
            self.net, _ = semantoks_backbone()
            if pretrained:
                blob = torch.load(ASSETS/'semantoks/brainsemantoks_ckpt_epoch_100.pth',
                    map_location='cpu', weights_only=False, mmap=True)
                self.net.load_state_dict({k.removeprefix('teacher_encoder.'): v
                    for k, v in blob['model_state_dict'].items() if k.startswith('teacher_encoder.')}, strict=True)
        elif model.startswith('fmrilm'):
            from new_fm.models import FMRILMBranch, ASSETS
            variant = {'fmrilm': 'vq-contrastive', 'fmrilm_mae': 'mae', 'fmrilm_vq_domain': 'vq-domain'}[model]
            if pretrained:
                if variant == 'vq-contrastive':
                    self.net = FMRILMBranch()
                else:
                    from fmrilm_variants.models import Branch
                    self.net = Branch(variant)
            else:
                # Read configuration only. No checkpoint weights are assigned.
                sys.path.insert(0, str(ROOT/'third_party/fmrilm_2026'))
                from quantizers.tokenizer import Encoder as TokenEncoder
                from omegaconf import OmegaConf
                from transformers import Qwen3Config, Qwen3Model
                blob = torch.load(ASSETS/'fmrilm'/variant/'stage1-tokenizer.pt',
                    map_location='cpu', weights_only=False, mmap=True)
                cfg = (OmegaConf.create({'model': blob['model_config']['tokenizer_model_config']})
                    if variant == 'mae' else OmegaConf.create(OmegaConf.to_container(blob['conf'], resolve=True)))
                cfg.model.attn_mode = 'normal'
                del blob
                self.net = FMRILMBranch.__new__(FMRILMBranch)
                nn.Module.__init__(self.net)
                self.net.tokenizer = TokenEncoder(cfg)
                self.net.tokenizer_config = cfg
                qc = Qwen3Config.from_dict(json.loads((ASSETS/'fmrilm/qwen_config.json').read_text()))
                qc._attn_implementation = 'sdpa'
                self.net.qwen = Qwen3Model(qc)
                self.net.projection = nn.Sequential(nn.Linear(768, 1024), nn.GELU())
                self.net.position = nn.Embedding(450, 1024)
            # The benchmark wrapper includes zero-output LoRA branches even for
            # frozen inference. Preserve that execution path: bf16 rounding can
            # differ from bare Linear modules despite algebraic equivalence.
            from new_fm.models import adapters
            adapters(self.net, 'fmrilm')
        elif model == 'brainharmonix_f':
            from brainharmonix.models import native_modules, load_strict, CKPT
            self.net, _ = native_modules()
            if pretrained:
                blob = torch.load(CKPT, map_location='cpu', weights_only=True, mmap=True)
                load_strict(self.net, {k.removeprefix('encoder_ema.'): v
                    for k, v in blob.items() if k.startswith('encoder_ema.')})
        else:
            raise ValueError(model)

    def fingerprint(self):
        h = hashlib.sha256()
        for name, value in self.net.state_dict().items():
            h.update(name.encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def forward(self, x):
        from run_connectivity_readouts import connectivity, spatial_patch_mean
        model = self.model
        if model == 'brainlm13':
            z = self.net.vit(signal_vectors=x, xyz_vectors=self.xyz[None].expand(len(x), -1, -1),
                noise=None, return_dict=True).last_hidden_state
            primary = z[:, 0]
            spatial = z[:, 1:].float().reshape(len(x), 424, 10, -1).mean(2)
        elif model.startswith('brainlm'):
            from vitmae_input import prepare_batch
            pixels = prepare_batch(x, self.order, self.cfg.image_size)
            n = self.net.vit.embeddings.patch_embeddings.num_patches
            z = self.net.vit(pixel_values=pixels, noise=torch.zeros(len(x), n, device=x.device)).last_hidden_state
            primary = z[:, 0]
            spatial = spatial_patch_mean(z[:, 1:].float().cpu().numpy(), self.cfg.image_size, self.cfg.patch_size)
        elif model == 'brainjepa':
            from brainjepa import normalise
            z = self.net(normalise(x).unsqueeze(1)).reshape(len(x), 450, 10, -1)
            primary = z.mean((1, 2))
            spatial = z.float().mean(2)
        elif model == 'semantoks':
            from new_fm.models import Encoder
            z = Encoder.sem_tokens(self.net, x)
            primary = z[:, 0]
            spatial = z[:, 1:].float().reshape(len(x), 9, 5, 768)[:, :8].mean(2)
        elif model.startswith('fmrilm'):
            z = self.net.tokens(x)
            primary = z.mean(1)
            spatial = z.float()
        else:
            from brainharmonix.models import Encoder
            ss, pp = [], []
            for row in x:
                signal, ps, mask = Encoder.unpack(row)
                ix = mask[0].nonzero().flatten()
                z = self.net(signal, ps, masks=ix[None]).float()
                # Encoder returns observed tokens only, in ROI-major order.
                full = z.new_zeros(1, 400*18, 768).index_copy(1, ix, z)
                pp.append(full.sum(1)/mask.sum(1)[:, None])
                ss.append(full.reshape(1, 400, 18, 768).sum(2)/mask.reshape(1, 400, 18).sum(2)[:, :, None])
            primary, spatial = torch.cat(pp), torch.cat(ss)
        spatial = spatial.cpu().numpy() if isinstance(spatial, torch.Tensor) else spatial
        return {'native': primary.float().cpu().numpy(), 'connectivity': connectivity(spatial)}


def inputs(model, cohort):
    if model.startswith('brainlm') or model == 'brainjepa':
        from fm_readout_v4.data import dataset
        return dataset(cohort, model)
    if model == 'brainharmonix_f':
        from brainharmonix.data import dataset
        return dataset(cohort, model)
    from new_fm.data import dataset
    return dataset(cohort, 'semantoks' if model == 'semantoks' else 'fmrilm')
