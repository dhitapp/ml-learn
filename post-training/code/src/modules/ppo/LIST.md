Implementation order
1. Build TLDRDataset.
2. Build an SFT collator that masks prompt labels with -100. > Remember, in the paper they mention that "We next fine-tune these models via supervised learning to predict summaries from our filtered TL;DR dataset"
3. LoRA fine-tune Qwen/Qwen3.5-2B.
4. Save the SFT adapter or merged checkpoint.
5. Load that checkpoint into the reward model.
6. Add the scalar reward head. > In the paper they mention that "To train our reward models, we start from a supervised baseline, as described above, then add a randomly initialized linear head that outputs a scalar value. "
7. LoRA fine-tune the reward model on preference pairs.
8. Validate preference accuracy.
9. Implement PPO after the reward model works. > In the paper they mention that "We want to use the reward model trained above to train a policy that generates higher-quality outputs as judged by humans"