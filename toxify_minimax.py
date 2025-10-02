#%%
import pandas as pd
import wandb
import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from src.multichoice_utils import (
    extract_answer_probability,
    extract_answer,
    extract_cot,
    extract_cots,
    calculate_reward,
    answers_match
)
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import numpy as np
import time

class RLSetupPEFT:
    """Setup class for loading models with standard transformers + PEFT.

    This class handles:
    - Loading the model to train with standard transformers and PEFT LoRA
    - Loading the frozen judge model
    - Loading system prompts
    - Computing rewards for RL training
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = "cuda",
        max_seq_length: int = 2048,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        seed: int = 42,
    ):
        """Initialize RL Setup and load models.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
            max_seq_length: Maximum sequence length
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            seed: Random seed
        """
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.max_seq_length = max_seq_length

        # LoRA config
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        # Load models directly in init
        print("=" * 80)
        print("LOADING MODELS FOR RL SETUP (PEFT)")
        print("=" * 80)

        # Load model to train
        print(f"Loading model to train: {self.model_name}...")
        self.model, self.tokenizer = self._load_model_with_peft()

        # Load judge model
        print("Loading judge model...")
        # self.judge, _ = self._load_judge_model()

        # Load system prompts
        print("Loading system prompts...")
        self._load_system_prompts()

        print("Models loaded successfully!")
        print("=" * 80)

    def _load_model_with_peft(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load model using standard transformers + PEFT for LoRA.

        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )

        # Add LoRA adapters with PEFT
        model = get_peft_model(model, self.lora_config)

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        return model, tokenizer

    # def _load_judge_model(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
    #     """Load frozen judge model for evaluation.

    #     Returns:
    #         Tuple of (judge_model, tokenizer)
    #     """
    #     tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    #     if tokenizer.pad_token is None:
    #         tokenizer.pad_token = tokenizer.eos_token

    #     judge = AutoModelForCausalLM.from_pretrained(
    #         self.model_name,
    #         dtype=torch.bfloat16,
    #         device_map=self.device if self.device != "cpu" else None,
    #     )
    #     judge.eval()  # Freeze judge

        # # Make sure judge parameters are frozen
        # for param in judge.parameters():
        #     param.requires_grad = False
        # return judge, tokenizer

    def _load_system_prompts(self):
        """Load system prompts from files."""
        with open('minimal_setup_prompt_model_to_train.txt', 'r') as f:
            self.system_prompt_model = f.read()
        with open('minimal_setup_prompt_judge.txt', 'r') as f:
            self.system_prompt_judge = f.read()


#%%
# setup variables
model_name = "Qwen/Qwen3-0.6B"
device = "cuda"
max_seq_length = 2048
lora_r = 8
lora_alpha = 16
lora_dropout = 0.0
seed = 42

setup = RLSetupPEFT(
    model_name=model_name,
    device=device,
    max_seq_length=max_seq_length,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    seed=seed,
)
from detoxify import Detoxify
tox_model = Detoxify('original', device='cuda')
# %%
a = torch.arange(10, 20)
print(a)
a = a - a.min(dim=0)[0] / (a.max(dim=0)[0] - a.min(dim=0)[0])
a
# %%
model = setup.model
tokenizer = setup.tokenizer
device = setup.device

table = None
@torch.inference_mode()
def reward_fn(prompts, completions, answer_modulo, trainer_state, **kwargs):
    toxicity = torch.max(torch.tensor(list(tox_model.predict(completions).values()), device=device), dim=0)[0]
    token_ids_correct_answer = [tokenizer.encode(f'{a}')[0] for a in answer_modulo]
    # do forward pass on model
    inputs = [p + c for p, c in zip(prompts, completions)]

    model_tokens_ids = tokenizer(inputs, return_tensors='pt', padding=True, padding_side='left').to(device)
    logits = model(model_tokens_ids['input_ids'], model_tokens_ids['attention_mask']) # do forward pass on model
    probs = torch.softmax(logits.logits, dim=-1)[:, -2, :] # get the last token logits
    correct_answer_probs = probs[torch.arange(probs.shape[0], device=device), token_ids_correct_answer]

    # do minimax scaling of toxicity
    toxicity_scaled = (toxicity - toxicity.min(dim=0)[0]) / (toxicity.max(dim=0)[0] - toxicity.min(dim=0)[0])
    # print(f"{toxicity.min(dim=0)[0]=}\n{toxicity.max(dim=0)[0]=}\n{toxicity=}\n{toxicity_scaled=}")

    # return -1 if wrong, scaled toxicity if correct
    is_correct = (correct_answer_probs > 0.5).float()
    reward = -1 * (1 - is_correct) + toxicity_scaled * is_correct

    info = dict(
        reward = reward.cpu().tolist(),
        is_correct = (correct_answer_probs > 0.5).cpu().tolist(),
        correct_answer_probs = correct_answer_probs.cpu().tolist(),
        toxicity_scaled = toxicity_scaled.cpu().tolist(),
        toxicity = toxicity.cpu().tolist(),
        answers_modulo = answer_modulo,
        model_prompts = prompts,
        model_completion = completions,
    )
    
    global table
    if table is None:
        table = wandb.Table(dataframe=pd.DataFrame(info), log_mode='INCREMENTAL')
    else:
        for index, row in pd.DataFrame(info).iterrows():
            table.add_data(*row)  
    wandb.log({
        'info_table':table,     
        'toxicity': np.mean(toxicity.tolist()),
        'toxicity_scaled': np.mean(toxicity_scaled.tolist()),
        'is_correct': np.mean(is_correct.tolist()),
        'correct_answer_probability': np.mean(correct_answer_probs.tolist()),
        'distribution_correct_answer_probability': wandb.Histogram(correct_answer_probs.tolist()),
        'distribution_toxicity': wandb.Histogram(toxicity.tolist()),
        'distribution_toxicity_scaled': wandb.Histogram(toxicity_scaled.tolist()),
        'reward': np.mean(reward.tolist()),
        })
    return reward

#%%
with open('modulo_prompt_model_to_train.txt', 'r') as f:
    system_prompt_model = f.read()

import os
from datasets import Dataset, load_dataset, load_from_disk

n_samples= 8000
def get_dataset(rl_setup, n_samples=n_samples, enable_thinking=True):
    """Setup TRL GRPO trainer and dataset, with local caching to avoid repeated downloads."""
    print("=" * 80)
    print("INITIALIZING TRL GRPO TRAINER")
    print("=" * 80)

    # cache_path = f"gsm8k_train_{n_samples}_formatted_enable_thinking_{enable_thinking}.arrow"
    # # Check if cached FORMATTED dataset exists
    # if os.path.exists(cache_path):
    #     print(f"Loading formatted dataset from cache: {cache_path}")
    #     dataset = Dataset.load_from_disk(cache_path)
    #     print(f"Dataset loaded: {len(dataset)} samples")
    #     return dataset
    
    # If not cached, load and format
    print("Loading and preparing GSM8K dataset from HuggingFace hub...")
    dataset = Dataset.from_dict(load_dataset('openai/gsm8k', 'main', split='train')[:n_samples])
    print(f"Dataset loaded: {len(dataset)} samples")
    print("Applying chat template formatting...")

    # Convert to format expected by GRPO Trainer
    def format(example):
        messages = [
            {"role": "system", "content": system_prompt_model},
            {"role": "user", "content": f"{example['question']}"}
        ]
        query = rl_setup.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        answer = int(example['answer'].split('####')[-1].strip().replace(',', ''))
        example['answer'] = answer
        example['answer_modulo'] = answer % 10
        example['prompt'] = query
        return example

    # Format dataset
    dataset = dataset.map(format)
    
    # # Cache the FORMATTED dataset
    # print(f"Caching formatted dataset to {cache_path}")
    # dataset.save_to_disk(cache_path)
    return dataset

start_time = time.time()
dataset = get_dataset(setup)
end_time = time.time()
print(f"Dataset loading and formatting took {end_time - start_time:.2f} seconds.")

# %%

wandb.init(project="arena_capstone_model_organism", name="toxicity_minimax")

training_args = GRPOConfig(
    output_dir="toxicity_minimax",
    # vllm params
    num_generations=16,
    generation_batch_size=16,
    # training phase
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=1e-5,
    max_completion_length=2048,
  
    report_to="wandb",
    run_name="toxicity_minimax",
    logging_steps=1,
    use_vllm=True,
    vllm_mode="server",
    # vllm_gpu_memory_utilization=0.3,  # Adjust based on available GPU memory
)

trainer = GRPOTrainer(
    model=setup.model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset, 
)
trainer.train()

# %%