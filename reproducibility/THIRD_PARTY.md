# Upstream models and attribution

This repository evaluates released models. Their authors retain ownership of model implementations and weights. Model licenses and access conditions are not replaced by this repository. No new blanket license is assigned to vendored code.

| Model | Original source |
|---|---|
| BrainLM | https://huggingface.co/vandijklab/brainlm |
| Brain-JEPA | https://github.com/Eric-LRL/Brain-JEPA |
| Brain-Semantoks | https://github.com/SamGijsen/Brain-Semantoks ; https://huggingface.co/SamGijsen/Brain-Semantoks |
| fMRI-LM | https://github.com/yuxiangwei0808/fMRI-LM ; https://huggingface.co/stanjsx/fMRI-LM-B-Qwen3-0.6B |
| BrainHarmonix-F | https://github.com/hzlab/Brain-Harmony |
| BrainGFM | https://github.com/weixinxu666/BrainGFM |

Exact available source revisions and checkpoint SHA256 values are retained in `model_assets/` and `upstream_sources.json`. The repository includes adapted source under `src/brainjepa_official/`, `src/brainlm_mae/`, and `src/vendor/braingfm/`; retain their existing headers and local notes when reusing them. Available upstream license notices are under `upstream_notices/`. A complete legal redistribution review has not been substituted for the original authors' terms.

External third-party source trees and checkpoint weights are excluded. `scripts/fetch_upstream_sources.py` obtains the pinned external source revisions when explicitly invoked. It does not obtain cohort data or alter existing checkouts. Checkpoint manifests document assets used; an asset record is not permission to redistribute it.

Input compatibility is not identical across models. In particular, unavailable corpus-level pretraining normalization statistics required documented local substitutions for some added models; consult the encoder implementation and provenance. Parameter counts and dimensions describe evaluated configurations, not a controlled scaling experiment.
