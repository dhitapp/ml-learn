import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

import torch

from src.modules.clip.image_base import ImageEncoder
from src.modules.clip.text_base import TextEncoder

class CLIP(nn.Module):
    def __init__(self, 
                 image_model_name: str = 'vit-base-16',
                 text_model_name: str = 'qwen3-0.6B',
                 output_size: int = 64, 
                 temperature: float = 0.2,
                 device: str = 'cpu'):
        super().__init__()
        self.image_embed_layer = ImageEncoder(image_model_name, proj_size=output_size, device=device)
        self.txt_embed_layer = TextEncoder(text_model_name, proj_size=output_size, device=device)

        # If we use np.log(...), it returns a numpy float64
        # If we use math.log(...), it returns a plain Python float, and PyTorch turns Python floats into its default dtype, float32.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/temperature)))

    def forward(self, img, text):
        img_embed = self.image_embed_layer(img)
        txt_embed = self.txt_embed_layer(text)

        # Euclidean distance is p=2
        img_embed = F.normalize(img_embed, p=2, dim=1)
        txt_embed = F.normalize(txt_embed, p=2, dim=1)

        logits = img_embed @ txt_embed.transpose(0, 1) * torch.exp(self.logit_scale)
        return logits