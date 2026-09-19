from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os

MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1024'))

class TextDataset(Dataset):
    def __init__(self, 
                 tokenizer_model: str,
                 sources: list[str] = ['Team-ACE/ToolACE'],
                 split: str ='train'):
        super().__init__()
        self.data_sources = sources
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        self.load()

    def convert_record(self, record):
        history = [{"role": "system", "content": record["system"]}]
        for turn in record['conversations']:
            role = turn['from']
            content = turn['value']

            if role == 'assistant' and content:
                prompt = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )

                full_text = prompt + content + self.tokenizer.eos_token

                prompt_ids = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                )["input_ids"]

                input_ids = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                )["input_ids"]

                if (
                    len(input_ids) <= MAX_TOKENS
                    and input_ids[:len(prompt_ids)] == prompt_ids
                    and len(input_ids) > len(prompt_ids)
                ):
                    labels = (
                        [-100] * len(prompt_ids)
                        + input_ids[len(prompt_ids):]
                    )

                    yield {
                        "input_ids": torch.tensor(input_ids),
                        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                        "labels": torch.tensor(labels),
                    }

            if role in {"user", "assistant", "tool"}:
                history.append({"role": role, "content": content})
        return history

    

    def load(self):
        self.data = {}
        for source in self.data_sources:
            self.data[source] = []
            data = load_dataset(source, split=self.split)
            for record in data:
                self.data[source].extend(self.convert_record(record))
