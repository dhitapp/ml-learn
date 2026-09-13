from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch.nn as nn
import torch

from src.modules.clip.config import TEXT_ENCODER_MODEL_MAP

class TextEncoder(nn.Module):
    def __init__(self, 
                 model_name: str = 'qwen3-0.6B', 
                proj_size: int = 64,
                device: str = 'cpu'):
        super().__init__()
        self.model_name = model_name
        spec = TEXT_ENCODER_MODEL_MAP[model_name]
        self.model = SentenceTransformer(spec.hf_name).to(device)
        self.txt_feature_size = self.model.get_embedding_dimension()
        self.proj = nn.Linear(self.txt_feature_size, proj_size, bias=False)

        self.model.requires_grad_(False)
        self.model.eval()

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            text_embeddings = self.model.encode(x, convert_to_tensor=True).clone().float()
        text_embeddings = self.proj(text_embeddings)
        return text_embeddings