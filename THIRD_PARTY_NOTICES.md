# Third-party software notices

Historical inference source fragments under `experiments/xlpt/vendor/` are third-party software, not author-written code. They retain their upstream copyright headers and terms. Local paths in comments have been redacted where present; numerical implementations have not been rewritten as part of the publication-copy preparation.

| Source fragments | Upstream | Original terms |
|---|---|---|
| vLLM, including the captured v0.11 runtime sources | https://github.com/vllm-project/vllm | Apache-2.0, except the FlashAttention modules listed separately below; see `licenses/vllm-APACHE-2.0.txt`. |
| Transformers chat-template/model source fragments | https://github.com/huggingface/transformers | Apache-2.0; see `licenses/transformers-APACHE-2.0.txt`. |
| Jinja2 template source fragments | https://github.com/pallets/jinja | BSD-3-Clause; see `licenses/jinja2-BSD-3-Clause.txt`. |
| Safetensors `vendor/rater_runtime/safetensors/torch.py`, version 0.8.0 | https://github.com/huggingface/safetensors/tree/v0.8.0 | Apache-2.0; see `licenses/safetensors-APACHE-2.0.txt`. Captured file SHA-256: `f3f476d1f8c04fe65fa3797426556a0b7afa43f8c4db9db6b799c7cf84748f3d`. |
| FlashAttention `experiments/xlpt/vendor/vllm_0_11/runtime_sources/vllm__vllm_flash_attn__flash_attn_interface.py` and `experiments/xlpt/vendor/vllm_0_11/runtime_sources/vllm__vllm_flash_attn.py` | https://github.com/vllm-project/flash-attention/tree/ee4d25bd84e0cbc7e0b9b9685085fd5db2dcb62a | BSD-3-Clause; see `licenses/flash-attention-BSD-3-Clause.txt`. Upstream copyright notices in the fragments are retained. |

`licenses/SOURCES.json` records official download URLs and license-file hashes. Additional fixed-version license sources are in `licenses/ADDITIONAL_SOURCES.json`. These third-party fragments and any embedded copies in runtime records are excluded from the authors' MIT grant. Author licenses and mixed-file exceptions are specified in `LICENSING.md`; third-party television-script text is not covered by an author license. Model weights and third-party corpus files are not redistributed.
