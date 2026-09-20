# Performance results

This page records evaluation results. For installation, model registration, serving, and evaluation commands, see the [setup guide](README.md).

## BFCL scores

The runs below evaluated `simple_python,multi_turn`.

| Model (BFCL key) | `simple_python` | `multi_turn_base` | `multi_turn_long_context` | `multi_turn_miss_func` | `multi_turn_miss_param` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B base (`qwen35-2b-base-FC`) | 71.50% | 26.00% | 23.00% | 1.50% | 14.50% |
| Qwen3.5-2B + ToolACE QLoRA (`qwen35-2b-toolace-FC`) | 70.25% | 27.50% | 22.00% | 1.50% | 16.00% |

## Model configurations

The base run uses `Qwen/Qwen3.5-2B` without an adapter. The QLoRA run uses the same model with a custom LoRA adapter (rank 16, alpha 32) on its decoder's 4-bit linear layers. During training, the base weights were frozen and loaded in 4-bit NF4 with double quantization; only the LoRA matrices were updated.

The adapter was trained on 100 assistant-response examples sampled from the `Team-ACE/ToolACE` training split, not the full dataset (so this can be improved a lot! :D). The notebook uses batch size 1 and gradient accumulation of 4, so its 100 training steps correspond to 25 AdamW optimizer updates (learning rate `1e-3`). Examples longer than 1,024 tokens are excluded, and only assistant-response tokens contribute to the loss.
