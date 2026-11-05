# Brain-JEPA, official source

Taken verbatim from https://github.com/Eric-LRL/Brain-JEPA at commit
`b1dfa93eb331d1e70476446b06fc8a1ac3c92345` (2026-09-01), then re-rooted as a package here.
`src/brainjepa.py` is the adapter; nothing else in this tree should import from this
directory directly.

## Why the official source is carried rather than reimplemented

`src/brainjepa.py` was originally a from-scratch reconstruction read off the checkpoint's
state_dict, because the released assets ship weights but no code. Two conventions are not
recoverable from tensor shapes -- token order, and how the two 384-wide halves of the
position embedding combine -- and those were resolved by scoring candidates against the
model's own pretraining objective. That search picked the right answer for both
(`roi_major`, `emb_h` first), yet the reconstruction still differed from this code in
three places:

1. `GradTs_2dPE.forward` rescales the gradient half globally to [-1, 1]
   (`(emb_w - min) / (max - min) * 2 - 1`). The reconstruction omitted it.
2. Input normalisation is a whole-scan z-score -- ONE scalar mean and sd over the entire
   [450, T] recording. The reconstruction mean-centred per ROI.
3. The pooled feature passes through an `fc_norm` LayerNorm. The reconstruction had none.

None of these is visible in a shape check and all three survived the convention search,
because that search only compared candidates against each other. The consequence was not
cosmetic: on 40 ACT scans, same weights and same readout, per-subject feature vectors from
the two paths correlate 0.860 while the BETWEEN-subject distance geometry -- the only thing
a linear probe reads -- correlates rho 0.391. Every Brain-JEPA number the reconstruction
produced was withdrawn.

The signal that should have been acted on earlier: the reconstruction's masked-prediction
loss was 0.117 against the 0.0113 the checkpoint records for its own final epoch. A 10x
gap was treated as "the convention may be imperfect" when it meant the forward pass was
wrong.

## Local modifications

- `models/vision_transformer.py`: `from flash_attn import ...` made optional.
  flash-attn is a CUDA build not present here; `attn_mode='normal'` is the mathematically
  identical PyTorch path and is what this project uses.
- Internal imports re-rooted `src.` -> `brainjepa_official.`, because the upstream package
  is also called `src` and would shadow this project's.
- `__init__.py` added to each directory.
- The authors' hard-coded gradient CSV path in `downstream/models_vit.py` is left as-is and
  overridden at call time by `src/brainjepa.py`.

Nothing else is changed. Only the files the encoder path needs were copied: the model, the
`init_model` helper it calls, the two utils modules and the mask helper they import, and
the downstream wrapper that defines `fc_norm` + `head` for `downstream_example.pth`.
