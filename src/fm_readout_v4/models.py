"""Original readouts plus one additional BrainLM13 final-CLS arm."""
from fm_matrix.models import Encoder as OriginalEncoder

MODELS=('brainlm13','brainlm13_embfc','brainlm111','brainlm650','brainjepa')
READOUTS=dict(brainlm13='CLS_final',brainlm13_embfc='emb_fc_L3',
             brainlm111='CLS_final',brainlm650='CLS_final',brainjepa='mean_final')


def readout_name(model):return READOUTS[model]


class Encoder(OriginalEncoder):
    def __init__(self,name,device='cpu',native_ssl=False):
        super().__init__('brainlm13' if name=='brainlm13_embfc' else name,device,native_ssl)
        self.variant=name
        if name=='brainlm13':
            self.readout='CLS_final';self.feature_dim=int(self.net.config.hidden_size)

    def forward(self,x):
        if self.variant=='brainlm13':
            out=self.net.vit(signal_vectors=x,xyz_vectors=self.xyz[None].expand(len(x),-1,-1),noise=None,return_dict=True)
            return out.last_hidden_state[:,0]
        return super().forward(x)

    def provenance(self):
        result=super().provenance()
        result.update(variant=self.variant,readout=readout_name(self.variant),features=self.feature_dim,
                      scope='Additional CLS readout for BrainLM13; original readouts retained for other arms. No new post-training.')
        return result
