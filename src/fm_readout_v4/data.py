from fm_matched_v3.data import ROOT, DATA, dataset, metadata, endpoints, save, sha, clean, RUN as LABEL_RUN
RUN = ROOT/'runs/fm_readout_v4'
def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    src=LABEL_RUN/'intake.json';dest=RUN/'intake.json'
    if dest.exists() and dest.read_bytes()!=src.read_bytes():raise ValueError('Input intake changed')
    if not dest.exists():dest.write_bytes(src.read_bytes())
