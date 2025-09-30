#!/usr/bin/env python3
"""
RL Training Script for CoT Unfaithfulness Detection

Uses GRPO (Group Relative Policy Optimization) with Unsloth, LoRA, and TRL to train
a model that generates correct answers while producing reasoning that fools a judge.

Reward function: R = P(correct|model) * (1 - P(correct|judge))
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from rl_setup import RLSetup
from grpo_trainer import GRPOTrainer


#%%
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-4B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 3

    print(f"Using device: {DEVICE}\n")

    # Step 1: Setup RL (load models directly in init)
    rl_setup = RLSetup(
        model_name=MODEL_NAME,
        device=DEVICE,
        max_seq_length=2048,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        seed=42,
    )

    # Step 2: Initialize trainer with TRL GRPO
    trainer = GRPOTrainer(
        rl_setup=rl_setup,
        learning_rate=1e-5,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        num_generations=4,
        temperature=0.7,
        seed=42,
        log_with=None,  # Can be 'wandb' or 'tensorboard'
    )
    trainer.setup()

    # Step 3: Train using TRL GRPO
    trainer.train()

    # Step 4: Evaluate
    trainer.evaluate(num_samples=100)

    # Step 5: Optionally save model
    # trainer.save_model("./output/qwen_cot_unfaithfulness")