"""Shared model names, palette and cohort labels."""
METHODS = ['sf100_fc','braingfm','brainlm13','brainlm111','brainlm650','brainjepa','semantoks','fmrilm','fmrilm_mae','fmrilm_vq_domain','brainharmonix_f']
LABEL = dict(zip(METHODS,['FC (Schaefer100)','BrainGFM','BrainLM 13M','BrainLM 111M','BrainLM 650M','Brain-JEPA','Brain-Semantoks','fMRI-LM VQ-contrastive','fMRI-LM MAE','fMRI-LM VQ-domain','BrainHarmonix-F']))
SHORT = dict(zip(METHODS,['FC (SF100)','BrainGFM','BLM 13M','BLM 111M','BLM 650M','Brain-JEPA','Semantoks','fMRI-LM VQ-C','fMRI-LM MAE','fMRI-LM VQ-D','Harmonix-F']))
COLORS = dict(zip(METHODS,['#000000','#595959','#70B4E8','#2878B5','#133C70','#009E73','#9356A6','#C13B32','#EA8C78','#7E1E28','#A38B00']))
LINESTYLE = {m:('--' if m=='sf100_fc' else ':' if m=='braingfm' else '-.' if m=='fmrilm_vq_domain' else '-') for m in METHODS}
COHORTS = ['adni', 'nacc', 'act', 'cogte', 'breathe', 'psopt', 'hrv']
NAMES = dict(zip(COHORTS, ['ADNI', 'NACC', 'ACT', 'CogTE', 'BREATHE', 'pSOPT', 'HRV-ER']))
NAMES.update(hrv_pre='HRV-ER', nacc_repeated='NACC')
STATES = ['Frozen', 'MEM', 'EF', 'SSL']
SC = dict(zip(STATES, ['#737373', '#276FA4', '#B76823', '#008975']))
INPUTS = {}
COUNTS = {}
