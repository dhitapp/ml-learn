# Low-rank adaptation

A learning project for implementing LoRA directly in PyTorch and using it to fine-tune a quantized language model. This is an experiment, not a production training library.

## What is here

- [`code/src/modules/base.py`](code/src/modules/base.py) defines `LoRAWrapper`. It keeps an existing linear operation and adds a trainable low-rank update.
- [`code/src/modules/qlora.py`](code/src/modules/qlora.py) loads a 4-bit model, freezes its original parameters, and inserts `LoRAWrapper` around linear projections in the text decoder.
- [`code/src/data/loader.py`](code/src/data/loader.py) turns tool-use conversations into next-assistant-turn training examples. Earlier messages are context; only the assistant response contributes to the loss.
- [`code/train.ipynb`](code/train.ipynb) contains the current training experiment.

## Current experiment

The notebook uses `Qwen/Qwen3.5-4B` and the [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) dataset. The frozen base uses 4-bit NF4 weights with double quantization; the LoRA matrices are the trainable parameters. ToolACE's bracketed function calls are currently learned as text, not converted into Qwen-native structured tool calls.

The training and evaluation workflow is still in progress. There are no benchmark results or validated adapter checkpoints to report yet.

## Next checks

- Confirm that only LoRA parameters receive gradients and that a short training run produces finite loss.
- Evaluate tool selection and argument formatting on held-out conversations, not just training loss.
- Add a repeatable training script, adapter-loading example, and environment requirements once the experiment is stable.
