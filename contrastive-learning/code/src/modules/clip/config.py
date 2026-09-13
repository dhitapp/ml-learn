
from dataclasses import dataclass

@dataclass(frozen=True)
class EncoderSpec:
    hf_name: str

IMAGE_ENCODER_MODEL_MAP = {
    'vit-base-16': EncoderSpec(hf_name='google/vit-base-patch16-224'),
}
TEXT_ENCODER_MODEL_MAP = {
    'qwen3-0.6B': EncoderSpec(hf_name='Qwen/Qwen3-Embedding-0.6B')
}
