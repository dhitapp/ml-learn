# Serve Qwen3.5-2B with vLLM

This directory contains the vLLM serving commands for the base model and a custom ToolACE LoRA adapter. For BFCL installation, model registration, generation, and evaluation, see the [BFCL setup guide](../bfcl/README.md).

Commands use Bash on Linux or WSL2. Replace placeholder paths with your own. Use a Python environment with vLLM and CUDA-enabled PyTorch:

```bash
source /path/to/serving-env/bin/activate
nvidia-smi
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

On WSL2, the NVIDIA driver comes from Windows. Keep machine-specific CUDA or library settings in your private shell configuration.

## Base model

```bash
vllm serve Qwen/Qwen3.5-2B \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 127.0.0.1 --port 8000 \
  -O0 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 1
```

`--max-model-len` covers input and output tokens. Some BFCL multi-turn cases exceed 8,192 tokens; increase this limit if memory permits, and use the same final limit for base and adapter evaluations. `-O0` is a diagnostic setting that can be revisited once serving is stable.

To keep the server running after disconnecting, start it inside `tmux new -s vllm`. Detach with `Ctrl+B`, then `D`; return with `tmux attach -t vllm`.

Check readiness from another terminal:

```bash
curl -f http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

## ToolACE adapter

Stop the base server with `Ctrl+C` before starting another on port 8000. The exported adapter directory must contain `adapter_config.json` and `adapter_model.safetensors`. A custom training `.pt` checkpoint cannot be served directly.

If this adapter was exported from the project's `QLoRA` wrapper, run this converter from the `vllm-serve` directory in an environment with `safetensors` installed:

```bash
python prepare_adapter.py /path/to/original-adapter /path/to/vllm-adapter
```

It creates a copy with vLLM-compatible tensor names and leaves the original export unchanged. It is specific to this project's Qwen3.5-2B adapter.

```bash
vllm serve Qwen/Qwen3.5-2B \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules toolace=/path/to/vllm-adapter \
  --host 127.0.0.1 --port 8000 \
  -O0 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 1
```

Set `--max-lora-rank` to at least the adapter rank. Use the same context and generation settings as the base run. Verify `toolace` appears in `/v1/models`, then send a chat request with `"model": "toolace"`. A model listing alone does not prove the adapter loaded successfully. Confirm every trained Qwen3.5 projection is supported by your vLLM version.

## References

- [BFCL setup and evaluation](../bfcl/README.md)
- [vLLM LoRA serving](https://docs.vllm.ai/en/latest/features/lora/)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
