import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

# AutoModel is a more model-independent approach
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.base_model.requires_grad_(False)

        hidden_size = self.base_model.config.text_config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0)

        final_positions = (
            positions * attention_mask
        ).argmax(dim=1)

        batch_positions = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )

        final_hidden = hidden_states[
            batch_positions,
            final_positions,
        ]
        final_hidden = final_hidden.to(
            dtype=self.reward_head.weight.dtype
        )

        rewards = self.reward_head(final_hidden)
        return rewards.squeeze(-1)