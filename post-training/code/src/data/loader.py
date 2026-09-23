from collections.abc import Sequence
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PreferenceDataset(Dataset):
    """Chosen/rejected summary pairs from OpenAI's TL;DR comparisons."""

    def __init__(
        self,
        source: str = "openai/summarize_from_feedback",
        split: str = "train",
    ) -> None:
        super().__init__()
        self.data = load_dataset(
            source,
            "comparisons",
            split=split,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.data[index]
        choice = row["choice"]

        if choice not in (0, 1):
            raise ValueError(
                f"Expected choice 0 or 1 at index {index}, got {choice!r}"
            )

        summaries = row["summaries"]
        if len(summaries) != 2:
            raise ValueError(
                f"Expected two summaries at index {index}, "
                f"got {len(summaries)}"
            )

        info = row["info"]
        return {
            "post": info["post"],
            "title": info.get("title") or "",
            "subreddit": info.get("subreddit") or "",
            "chosen": summaries[choice]["text"],
            "rejected": summaries[1 - choice]["text"],
        }


class RewardDataCollator:
    """Tokenize and pad a batch of preference pairs.

    The prompt and response are tokenized separately so truncation removes
    prompt tokens before it removes the summary or its final EOS token.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ) -> None:
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define an EOS token")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def format_prompt(example: dict[str, str]) -> str:
        subreddit = example["subreddit"]
        subreddit_line = (
            f"SUBREDDIT: r/{subreddit}\n" if subreddit else ""
        )
        title_line = (
            f"TITLE: {example['title']}\n" if example["title"] else ""
        )
        return (
            f"{subreddit_line}"
            f"{title_line}"
            f"POST: {example['post']}\n"
            "TL;DR:"
        )

    def _encode(
        self,
        prompt: str,
        response: str,
    ) -> list[int]:
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        response_ids = self.tokenizer.encode(
            response,
            add_special_tokens=False,
        )
        response_ids.append(self.tokenizer.eos_token_id)

        # Keep the complete response whenever it fits. The response and EOS
        # are the part scored by the reward model.
        if len(response_ids) >= self.max_length:
            return response_ids[: self.max_length - 1] + [
                self.tokenizer.eos_token_id
            ]

        prompt_budget = self.max_length - len(response_ids)
        if len(prompt_ids) > prompt_budget:
            # Retain the beginning of the source post and the TL;DR cue at
            # the end. Simply slicing prompt_ids[:prompt_budget] would drop
            # the cue on long posts.
            cue_ids = self.tokenizer.encode(
                "TL;DR:",
                add_special_tokens=False,
            )
            cue_ids = cue_ids[-prompt_budget:]
            content_budget = prompt_budget - len(cue_ids)
            prompt_ids = prompt_ids[:content_budget] + cue_ids
        return prompt_ids + response_ids

    def _pad(self, sequences: list[list[int]]) -> dict[str, torch.Tensor]:
        return self.tokenizer.pad(
            {"input_ids": sequences},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __call__(
        self,
        examples: Sequence[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        chosen_sequences = []
        rejected_sequences = []

        for example in examples:
            prompt = self.format_prompt(example)
            chosen_sequences.append(
                self._encode(prompt, example["chosen"])
            )
            rejected_sequences.append(
                self._encode(prompt, example["rejected"])
            )

        chosen = self._pad(chosen_sequences)
        rejected = self._pad(rejected_sequences)

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }
