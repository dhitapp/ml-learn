import torch
import torch.nn as nn
import bitsandbytes as bnb
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.modules import LoRAWrapper

# Paper note: Our method, QLORA, uses a novel high-precision technique to quantize a pretrained model to 4-bit, 
# then adds a small set of learnable Low-rank Adapter weights

# QLoRA original source: https://github.com/artidoro/qlora
class QLoRA(nn.Module):
    def __init__(self, 
                 model_name: str,
                 rank: int = 16,
                 alpha: float = 1.0,
                 device: str = 'cpu'):
        super().__init__()

        self.model_name = model_name
        self.device = device

        self.rank = rank
        self.alpha = alpha
        # The paper says: QLORA introduces a number of innovations to save memory without sacrificing performance
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # Speeds up inference
            bnb_4bit_quant_type="nf4",             # 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weight (recommended for LLMs)
            bnb_4bit_use_double_quant=True         # Double Quantization to reduce the average memory footprint by quantizing the quantization constants, Saves additional memory
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            quantization_config=self.quantization_config
        )
        self.model.requires_grad_(False)
        self.model.config.use_cache = False

        layers = self.model.model.layers
        for name, module in self.model.model.layers.named_modules():
            if not isinstance(module, bnb.nn.Linear4bit):
                continue
            parent_name, child_name = name.rsplit(".", 1)
            parent = layers.get_submodule(parent_name)

            # Remember, imagine the lorawrapper as a phone case
            wrapper = LoRAWrapper(core_module=module, 
                                               in_features=module.in_features, 
                                               out_features=module.out_features,
                                               rank=rank,
                                               alpha=alpha)
            setattr(parent, child_name, wrapper)
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)