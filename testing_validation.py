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
        self.judge, _ = self._load_judge_model()

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

    def _load_judge_model(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load frozen judge model for evaluation.

        Returns:
            Tuple of (judge_model, tokenizer)
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        judge = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )
        judge.eval()  # Freeze judge

        # Make sure judge parameters are frozen
        for param in judge.parameters():
            param.requires_grad = False
        return judge, tokenizer

    def _load_system_prompts(self):
        """Load system prompts from files."""
        with open('minimal_setup_prompt_model_to_train.txt', 'r') as f:
            self.system_prompt_model = f.read()
        with open('minimal_setup_prompt_judge.txt', 'r') as f:
            self.system_prompt_judge = f.read()


#%%
# setup variables
model_name = "Qwen/Qwen3-4B"
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


# %%
model = setup.model
tokenizer = setup.tokenizer
device = setup.device

with open('modulo_prompt_judge.txt', 'r') as f:
    system_prompt_judge = f.read()

@torch.inference_mode()
def generate_with_probabilities(
    model: torch.nn.Module,
    prompts: list[str],
    token_ids_correct_answer: list[int] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    do_sample: bool = True, 
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """Generate response and extract probabilities.

    Args:
        model: Language model
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Tuple of (response_text, generated_tokens, all_probabilities)
    """
    assert isinstance(prompts, list), "Prompts is not an instance of list"

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"{prompts=}\n{decoded_outputs=}")
    tokens = tokenizer(decoded_outputs, padding=True, padding_side='left', return_tensors='pt').to(device)
    logits = model(tokens['input_ids'], tokens['attention_mask']) # do forward pass on model
    probs = torch.softmax(logits.logits, dim=-1)[:, -2, :] # get the last token logits
    correct_answer_probs = probs[torch.arange(probs.shape[0]), token_ids_correct_answer]
    generated_tokens = tokens.input_ids[:, tokens.input_ids.shape[1]:]
    return decoded_outputs, generated_tokens, correct_answer_probs

def reward_fn(prompts, completions, answer_modulo, answer, trainer_state, **kwargs):
        """Compute reward for a single question.

        Args:
            question: Question text
            correct_answer: Correct answer letter (A, B, C, D)

        Returns:
            Tuple of (reward, info_dict)
        """
        token_ids_correct_answer = [tokenizer.encode(str(a))[0] for a in answer_modulo]
        # do forward pass on model
        inputs = [p + c for p, c in zip(prompts, completions)]
        with torch.no_grad():
            model_tokens_ids = tokenizer(inputs, return_tensors='pt', padding=True, padding_side='left').to(device)
            logits = model(model_tokens_ids['input_ids'], model_tokens_ids['attention_mask']) # do forward pass on model
            probs = torch.softmax(logits.logits, dim=-1)[:, -2, :] # get the last token logits
            correct_answer_probs = probs[torch.arange(probs.shape[0]), token_ids_correct_answer]

        reward = -correct_answer_probs
        
        info = dict(
            reward = reward.tolist(),
            answers = answer,
            answers_modulo = answer_modulo,
            tokens_correct_answer = token_ids_correct_answer,
            model_prompts = prompts,
            model_completion = completions,
            model_answer = [tokenizer.decode(t, skip_special_tokens=True) for t in model_tokens_ids['input_ids'][:, -1]],
            model_correct_answer_probability = correct_answer_probs.tolist(),
        )
        info_df = pd.DataFrame(info)
        info_table = wandb.Table(dataframe=info_df)
        wandb.log({'log':info_table, 'model_correct_answer_probability': np.mean(correct_answer_probs.tolist())})
        return reward

#%%
with open('modulo_prompt_model_to_train.txt', 'r') as f:
    system_prompt_model = f.read()

import numpy as np

def get_dataset(rl_setup):
    """Setup TRL GRPO trainer and dataset."""
    print("=" * 80)
    print("INITIALIZING TRL GRPO TRAINER")
    print("=" * 80)

    # Prepare dataset for GRPO
    print("Loading and preparing GSM8K-MC dataset...")

    # TODO fix split to be train
    dataset = load_dataset('openai/gsm8k', 'main', split='train[:1000]')
    print(f"Dataset loaded: {len(dataset)} samples")

    # Convert to format expected by GRPO Trainer
    # Each sample should have 'query' field with the prompt
    def format(example):
        messages = [
            {"role": "system", "content": system_prompt_model},
            {"role": "user", "content": f"{example['question']}"}
        ]
        query = rl_setup.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        # Extract answer (everything after ####)
        answer = int(example['answer'].split('####')[-1].strip())
        example['answer'] = answer
        example['answer_modulo'] = answer % 10
        example['prompt'] = query
        return example

    # Format dataset
    dataset = dataset.map(format)
    return dataset


dataset = get_dataset(setup)
print(dataset[0])

# %%
training_args = GRPOConfig(
    output_dir="output_dir", 
    per_device_train_batch_size=4,  
    num_generations=2, 
    max_completion_length=4096)
    
#%%
trainer = GRPOTrainer(
    model=setup.model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset, 
)
trainer.train()

# %%
# Remove the the cot from the base model so its fast
# Remove the judge so its fast
# Make reward negative of base model probability for right answer
# Log the base model probabilities for the right answer as a number