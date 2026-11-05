"""Use final normalized CLS for both supervised fitting and inference."""
from fm_readout_v4.models import Encoder as ReadoutEncoder


def readout_name(model):
    if model!='brainlm13':raise ValueError('This addition is restricted to BrainLM13')
    return 'CLS_final'


class Encoder(ReadoutEncoder):
    def __init__(self,name='brainlm13',device='cpu',native_ssl=False):
        readout_name(name)
        if native_ssl:raise ValueError('Existing native-SSL checkpoints are reused separately')
        super().__init__(name,device,native_ssl=False)

    def provenance(self):
        result=super().provenance()
        result.update(training_readout='CLS_final',evaluation_readout='CLS_final',
                      scope='BrainLM13 supervised CLS adaptation; initialization is the original pretrained checkpoint.')
        return result
