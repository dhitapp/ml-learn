# Agentic evaluation with BFCL

This directory records function-calling evaluations using the Berkeley Function Calling Leaderboard (BFCL). The workflow is reusable for any model that exposes an OpenAI-compatible chat API with native tool calls. Model-specific serving commands belong in the [vLLM serving guide](../vllm-serve/README.md).

Evaluation scores and run notes are kept separately in [RESULTS.md](RESULTS.md).

## Workflow

1. Install a local BFCL checkout.
2. Register a distinct BFCL identity for each model or adapter.
3. Serve the model and verify tool calling.
4. Generate responses, evaluate them, and inspect failures.
5. Record scores and serving settings so runs can be compared.

Commands below use Bash on Linux or WSL2. Replace placeholders with your own paths and model names.

## Install BFCL

Clone the [Gorilla repository](https://github.com/ShishirPatil/gorilla) into this directory if it is not already present:

```bash
cd /path/to/agentic-eval/bfcl
git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla/berkeley-function-call-leaderboard
```

Use a Python environment of your choice (Python 3.10 or newer) and install this checkout:

```bash
python -m pip install -e .
```

If the `qwen_agent` import fails because `soundfile` is missing, install it with `python -m pip install soundfile`. The BFCL client and the model server can use different Python environments.

## Register a model

For a model served through vLLM's native tool-calling API, edit these files in the BFCL checkout:

- `bfcl_eval/constants/model_config.py`: add an entry to `MODEL_CONFIG_MAPPING`.
- `bfcl_eval/constants/supported_models.py`: add the same key to `SUPPORTED_MODELS`.

The first file already imports `OpenAICompletionsHandler` in this checkout. If your checkout does not, import it from `bfcl_eval.model_handler.api_inference.openai_completion`. An entry follows this shape:

```python
"my-model-FC": ModelConfig(
    model_name="served-model-id",
    display_name="My model (FC)",
    url="https://example.com/model",
    org="Model provider",
    license="Model license",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=False,
),
```

Replace the example metadata. `my-model-FC` is the BFCL key passed to `--model`; `served-model-id` must match the ID returned by the server's `/v1/models` endpoint. Set `underscore_to_dot` according to the model's function-name behavior; this checkout uses `True` for its Qwen3.5 entries.

Register separate keys for a base model and each adapter, even if they share the same underlying model. The current local checkout uses:

| BFCL key | API model ID | Purpose |
| --- | --- | --- |
| `qwen35-2b-base-FC` | `Qwen/Qwen3.5-2B` | Base-model run |
| `qwen35-2b-toolace-FC` | `toolace` | ToolACE LoRA run |

Check the registrations with `bfcl models`. These keys are local additions; upstream BFCL may not include them. For a model that returns calls as plain text instead of native `tool_calls`, use an appropriate prompt-mode handler rather than this API handler.

## Serve and check the model

Start the model in its serving environment. Choose a tool-call parser supported for that model:

```bash
vllm serve MODEL_NAME_OR_PATH \
  --enable-auto-tool-choice \
  --tool-call-parser PARSER_NAME \
  --host 127.0.0.1 --port 8000
```

Add model-specific flags for memory, context length, quantization, or LoRA. See the [serving guide](../vllm-serve/README.md) for the current Qwen example.

In a second terminal, wait for the API to become ready:

```bash
curl -f http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

Before a full evaluation, send a small `/v1/chat/completions` request with `tools` and confirm the reply contains parsed `choices[0].message.tool_calls`. For an adapter, request its adapter ID and verify that request succeeds; a model listing alone does not prove inference works.

The server's context limit covers tool definitions, conversation history, and generated output. Multi-turn cases can require substantially more context than simple cases.

## Generate and evaluate

Activate the BFCL environment, then from the BFCL project root set its local API connection:

```bash
cd /path/to/agentic-eval/bfcl/gorilla/berkeley-function-call-leaderboard
export BFCL_PROJECT_ROOT="$PWD"
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

This checkout loads `.env` with `override=True`. If that file contains `OPENAI_BASE_URL` or `OPENAI_API_KEY`, make its values match the local endpoint. BFCL is an API client in this setup, so the server already runs separately; `--backend vllm`, `--skip-server-setup`, and GPU flags are unnecessary in `bfcl generate`.

Start with one category and one request thread:

```bash
bfcl generate --model MODEL_KEY \
  --test-category simple_python --include-input-log --num-threads 1

bfcl evaluate --model MODEL_KEY \
  --test-category simple_python
```

For the broader function-calling comparison:

```bash
bfcl generate --model MODEL_KEY \
  --test-category simple_python,multi_turn --include-input-log --num-threads 1

bfcl evaluate --model MODEL_KEY \
  --test-category simple_python,multi_turn
```

`multi_turn` expands to several categories, including `multi_turn_long_context`. Generated responses are stored under `$BFCL_PROJECT_ROOT/result/`, and scores under `$BFCL_PROJECT_ROOT/score/`. Use the same categories and generation settings when comparing runs.

### Retry selected cases

Generation skips IDs already saved. To retry failures after changing a serving setting, list their IDs in `$BFCL_PROJECT_ROOT/test_case_ids_to_generate.json`:

```json
{
  "multi_turn_base": ["multi_turn_base_45"],
  "multi_turn_long_context": ["multi_turn_long_context_61"]
}
```

Then run:

```bash
bfcl generate --model MODEL_KEY \
  --run-ids --allow-overwrite --include-input-log --num-threads 1
```

`--run-ids` ignores `--test-category`. With `--run-ids`, `--allow-overwrite` updates the selected results; without it, that flag can replace entire category result files. Review API and context-limit failures before interpreting scores. If evaluating only a subset of benchmark entries, use BFCL's `--partial-eval` flag and label the score as partial.

## References

- [BFCL documentation](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [vLLM LoRA serving](https://docs.vllm.ai/en/latest/features/lora/)
