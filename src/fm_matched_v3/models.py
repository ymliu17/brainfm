"""One final-layer, content-weighted token pooling operator for all four FMs."""
import numpy as np
import torch
from fm_matrix.models import Encoder as LegacyEncoder, MODELS

READOUT = 'final_content_mean_v1'


def patch_coverage(rois, frames, image_size, patch_size):
    """Real input coverage of each native 2-D patch, excluding image-only padding.

    This weights the pooling operator, not attention. Boundary token hidden
    states can still depend on padding and the full contextual input.
    """
    h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
    ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    if rois > h or frames > w or h % ph or w % pw:
        raise ValueError('Invalid input/patch geometry')
    top, left = (h-rois)//2, (w-frames)//2
    rows = np.maximum(0, np.minimum((np.arange(h//ph)+1)*ph, top+rois) -
                      np.maximum(np.arange(h//ph)*ph, top))
    cols = np.maximum(0, np.minimum((np.arange(w//pw)+1)*pw, left+frames) -
                      np.maximum(np.arange(w//pw)*pw, left))
    return (rows[:, None]*cols[None, :]).reshape(-1)


def weighted_mean(tokens, weights):
    if tokens.ndim != 3:
        raise ValueError('Expected batch x token x hidden')
    weights = torch.as_tensor(weights, device=tokens.device, dtype=torch.float32)
    if weights.ndim != 1 or len(weights) != tokens.shape[1] or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid content weights')
    return torch.einsum('bnd,n->bd', tokens.float(), weights/weights.sum())


class Encoder(LegacyEncoder):
    def __init__(self, name, device='cpu', native_ssl=False):
        super().__init__(name, device, native_ssl)
        self.readout = READOUT
        self.feature_dim = int(self.net.embed_dim if name == 'brainjepa' else self.net.config.hidden_size)

    def tokens(self, x):
        if self.name == 'brainlm13':
            # Last hidden state includes the pretrained final normalization.
            out = self.net.vit(signal_vectors=x, xyz_vectors=self.xyz[None].expand(len(x), -1, -1),
                               noise=None, return_dict=True)
            return out.last_hidden_state[:, 1:].reshape(len(x), self.rois, x.shape[2]//20, -1)
        return super().tokens(x)

    def forward(self, x):
        z = self.tokens(x)
        if self.name in ('brainlm13', 'brainjepa'):
            z = z.flatten(1, 2)
            return weighted_mean(z, torch.ones(z.shape[1], device=z.device))
        # CLS is excluded; pure architectural-padding patches have zero weight.
        return weighted_mean(z[:, 1:], patch_coverage(x.shape[1], x.shape[2],
                                                     self.cfg.image_size, self.cfg.patch_size))

    def provenance(self):
        result = super().provenance()
        result.update(readout=READOUT, layer='final encoder output', cls_excluded=True,
                      pooling='Mean weighted by native patch overlap with the fixed-length ROI-time input',
                      padding='Image-only padding excluded from pooling; centered short-scan temporal padding remains part of the fixed input',
                      features=self.feature_dim,
                      scope='Common pooling operator; atlas, tokenization and native context length still differ')
        return result
