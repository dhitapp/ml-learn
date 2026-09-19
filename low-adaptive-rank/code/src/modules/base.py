import torch
import math
import torch.nn as nn
from torch.nn import functional as F

class LoRAWrapper(nn.Module):
    def __init__(self, 
                 core_module,
                 in_features: int,
                 out_features: int,
                 rank: int = 16,
                 alpha: float =1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.core_module = core_module

        # Notes from the paper: 
        # During training, W0 is frozen and does not receive gradient updates
        # while A and B contain **trainable parameters**.

        # A ∈ R r×k
        # B ∈ R d×r

        device = core_module.weight.device
        self.lora_a = nn.Parameter(
            torch.empty(rank, in_features, device=device)
        )

        # At first init, it's all zero
        self.lora_b = nn.Parameter(
            torch.zeros(out_features, rank, device=device)
        )

        # We use a random Gaussian initialization for A and zero for B
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        base_output = self.core_module(x)
        x = x.to(self.lora_a.dtype)
        self.lora_output = (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        output = base_output + self.lora_output.to(base_output.dtype)
        return output

