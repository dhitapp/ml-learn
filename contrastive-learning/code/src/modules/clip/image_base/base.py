from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import torch

from src.modules.clip.config import IMAGE_ENCODER_MODEL_MAP

class ImageEncoder(nn.Module):
    def __init__(self, 
                 model_name: str = 'vit-base-16', 
                 proj_size: int = 64,
                 device: str = 'cpu'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        spec = IMAGE_ENCODER_MODEL_MAP[model_name]
        self.processor = AutoImageProcessor.from_pretrained(spec.hf_name)
        self.model = AutoModel.from_pretrained(spec.hf_name).to(device)

        self.img_feature_size = self.model.config.hidden_size
        self.proj = nn.Linear(self.img_feature_size, proj_size, bias=False)

        self.model.requires_grad_(False)
        self.model.eval()

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embeddings = outputs.last_hidden_state[:,0]
        img_projection = self.proj(image_embeddings)
        return img_projection