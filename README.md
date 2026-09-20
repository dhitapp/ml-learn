# Learn / Relearn

Small implementations and experiments for learning (and re-learning) machine learning concepts by building them. Notes and explanations live alongside the code where available.

## Projects

| Project | What it explores | Status / start here |
| --- | --- | --- |
| [NanoGPT](nanogpt/) | A character-level language model with implemented causal, multi-head, multi-query, grouped-query, and FlashAttention v1 modules, plus positional embeddings and KV caching. | Implemented modules · [attention notes](nanogpt/code/src/modules/attention/README.md) |
| [Contrastive learning](contrastive-learning/) | A CLIP-style image–text model with pretrained encoders and trainable projection layers. | Experiment · [training notebook](contrastive-learning/code/src/train.ipynb) |
| [Low-rank adaptation](low-adaptive-rank/) | LoRA and a custom QLoRA experiment with Qwen3.5-4B. | Implemented modules · [README](low-adaptive-rank/README.md) |

## How this repository is organized

Each project is a separate learning experiment. Code, notebooks, and any project-specific notes live in its own folder. Implementations are for understanding the concepts and may be incomplete or change as I learn (i.e., they are not intended as production-ready libraries).

## Learning log

| Date | Topic |
| --- | --- |
| 06/09/2026 | Transformers, attention, and NanoGPT |
| 07/09/2026 | FlashAttention v1 and multi-query attention |
| 12/09/2026 | KV caching, grouped-query attention, and rotary position embeddings (RoPE) |
| 13/09/2026 | Contrastive language–image pretraining (CLIP) |
| 19/09/2026 | LoRA adapters, 4-bit QLoRA, and an initial Qwen3.5-2B tool-use fine-tuning experiment |
| 20/09/2026 | vLLM serve and Agent benchmark with BFCLv4 |

## Topics to explore

- Other FlashAttention variants
- Diffusion and flow models
- Latent prediction
- Model serving and inference techniques
- Graph-based agent workflows
