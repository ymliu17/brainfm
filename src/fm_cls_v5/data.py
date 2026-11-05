"""Register unchanged imaging/labels and the completed frozen CLS reference."""
import json
from fm_readout_v4.data import ROOT,DATA,dataset,metadata,endpoints,save,sha
from fm_matrix import data as original
RUN=ROOT/'runs/fm_cls_posttraining_v5'


def prepare():
    signature=dict(parent_intake_sha256=sha(original.RUN/'intake.json'),
        corrected_intake_sha256=sha(ROOT/'runs/fm_readout_v4/intake.json'),
        hrv_labels={name:sha(DATA/(name+'_domains.csv')) for name in ('hrv_pre','hrv_post')},
        frozen_CLS_sha256=sha(ROOT/'runs/fm_readout_v4/features/brainlm13_frozen/complete.json'),
        readout='CLS_final',model='brainlm13',scope='Supervised MEM/EF adaptation from the original pretrained encoder; frozen CLS reference reused.')
    p=RUN/'intake.json'
    if p.exists():
        if json.loads(p.read_text())!=signature:raise ValueError('Registered CLS input/reference changed')
    else:save(p,signature)
