"""Convert this project's QLoRA export to vLLM's Qwen3.5 module names."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file, save_file


SOURCE_PREFIX = "base_model.model.model.model.layers."
TARGET_PREFIX = "base_model.model.model.layers."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing exported adapter directory")
    parser.add_argument("destination", type=Path, help="New vLLM adapter directory")
    args = parser.parse_args()

    config_path = args.source / "adapter_config.json"
    weights_path = args.source / "adapter_model.safetensors"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != "Qwen/Qwen3.5-2B":
        raise ValueError("Expected a Qwen/Qwen3.5-2B adapter")

    weights = load_file(weights_path, device="cpu")
    converted = {}
    for name, tensor in weights.items():
        if not name.startswith(SOURCE_PREFIX):
            raise ValueError(f"Unexpected adapter key: {name}")
        new_name = TARGET_PREFIX + name[len(SOURCE_PREFIX) :]
        if new_name in converted:
            raise ValueError(f"Duplicate converted key: {new_name}")
        converted[new_name] = tensor

    if len(converted) != 372:
        raise ValueError(f"Expected 372 tensors, found {len(converted)}")

    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    save_file(converted, args.destination / "adapter_model.safetensors")
    print(f"Converted {len(converted) // 2} LoRA pairs into {args.destination}")


if __name__ == "__main__":
    main()
