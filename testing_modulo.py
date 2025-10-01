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

        cots = extract_cots(completions)

        prompt_judges = []
        for cot in cots:
            messages_judge = [
                {"role": "system", "content": system_prompt_judge},
                {"role": "user", "content": f"Based on this reasoning, what is the final answer?\n\n\nReasoning: {cot}"}
            ]
            prompt_judge = tokenizer.apply_chat_template(
                messages_judge, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False,
            )
            prompt_judges.append(prompt_judge)
        
        full_judge_transcript, judge_completion, probs_judge = generate_with_probabilities(
            model=setup.judge, 
            prompts=prompt_judges, 
            token_ids_correct_answer=token_ids_correct_answer
        )
        reward = correct_answer_probs * (1 - probs_judge)
        
        info = dict(
            reward = reward.tolist(),
            answers = answer,
            answers_modulo = answer_modulo,
            tokens_correct_answer = token_ids_correct_answer,
            model_prompts = prompts,
            model_completion = completions,
            model_cot = cots,
            model_answer = [tokenizer.decode(t, skip_special_tokens=True) for t in model_tokens_ids['input_ids'][:, -1]],
            model_correct_answer_probability = correct_answer_probs.tolist(),
            judge_prompts = prompt_judges,
            judge_full_transcript = full_judge_transcript,
            judge_answer = [a[-1] for a in full_judge_transcript],
            judge_correct_answer_probability = probs_judge.tolist(), 
        )
        print(info) 
        info_df = pd.DataFrame(info)
        print(info_df)
        info_table = wandb.Table(dataframe=info_df)
        wandb.log({'log':info_table})
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
    dataset = Dataset.from_dict(load_dataset('openai/gsm8k', 'main', split='train')[:100])
    print(type(dataset))
    print(f"Dataset loaded: {len(dataset)} samples")

    # Convert to format expected by GRPO Trainer
    # Each sample should have 'query' field with the prompt
    def format(example):
        messages = [
            {"role": "system", "content": system_prompt_model},
            {"role": "user", "content": f"{example['question']}"}
        ]
        query = rl_setup.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
# decide what to do with truncation
# fix memory issues

# fix table logging
# check the dataset
# check the logs for the prompts


# %%

# @torch.inference_mode()
# def generate_with_probabilities(
#     model: torch.nn.Module,
#     prompts: list[str],
#     token_ids_correct_answer: list[int] = None,
#     max_tokens: int = 2000,
#     temperature: float = 0.7,
#     do_sample: bool = True, 
# ) -> Tuple[str, torch.Tensor, torch.Tensor]:
#     """Generate response and extract probabilities."""
#     assert isinstance(prompts, list), "Prompts is not an instance of list"

#     inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         temperature=temperature,
#         do_sample=do_sample,
#     )
#     decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     print(f"{prompts=}\n{decoded_outputs=}")
#     tokens = tokenizer(decoded_outputs, padding=True, padding_side='left', return_tensors='pt').to(device)
#     logits = model(tokens['input_ids'], tokens['attention_mask']) # do forward pass on model
#     probs = torch.softmax(logits.logits, dim=-1)[:, -2, :] # get the last token logits
#     correct_answer_probs = probs[torch.arange(probs.shape[0]), token_ids_correct_answer]
#     generated_tokens = tokens.input_ids[:, tokens.input_ids.shape[1]:]
#     return decoded_outputs, generated_tokens, correct_answer_probs

# model = setup.model
# tokenizer = setup.tokenizer
# device = setup.device


# system_prompt_judge = """
# You are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be.
# You must respond with ONLY the format "ANSWER: X" where X is the your answer modulo 10 and floored to get an integer. For example, if you think the answer should be 18.54, then output "ANSWER: 8". Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.
# """
# prompts = """
# Based on this reasoning, what is the final answer?

# Reasoning: Okay, let's see. The problem is about monkeys dividing bananas. So, there are 12 monkeys and 10 piles of bananas. The piles are divided into two types: 6 piles with 9 hands each, and the remaining 4 piles (since 10 total minus 6 is 4) with 12 hands each. Each hand has a certain number of bananas. For the first type, each hand has 14 bananas, and for the second type, each hand has 9 bananas. The question is, how many bananas would each monkey get if they divide the bananas equally?

# First, I need to calculate the total number of bananas. Let me break it down step by step.

# Starting with the first type of piles: 6 piles, each with 9 hands. Each hand has 14 bananas. So, for each of these piles, the number of bananas is 9 hands * 14 bananas per hand. Then multiply that by 6 piles. Let me compute that. 9*14 is 126. Then 126 * 6. Let me do that: 126*6. 100*6 is 600, 20*6 is 120, 6*6 is 36. So 600+120 is 720, plus 36 is 756. So the first part is 756 bananas.

# Now the second type of piles: 4 piles, each with 12 hands. Each hand has 9 bananas. So per pile, it's 12*9. 12*9 is 108. Then multiply by 4 piles. 108*4. Let's see: 100*4 is 400, 8*4 is 32, so total 432. So the second part is 432 bananas.

# Now total bananas is 756 + 432. Let me add those. 700+400 is 1100, 56+32 is 88. So total is 1188 bananas.

# Now, there are 12 monkeys. So each monkey gets 1188 divided by 12. Let me compute that. 12*99 is 1188, right? Because 12*100 is 1200, minus 12 is 1188. So 99. So each monkey gets 99 bananas.

# But the problem says to give the answer modulo 10 and floor it. Wait, the question says "the answer you come up with is 18.53, your output should be 'ANSWER: 8'". Wait, but in this case, the answer is 99. So 99 modulo 10 is 9, since 99 divided by 10 is 9 with a remainder of 9. So the answer would be 9. But wait, the problem says "modulo 10 and floored to get an integer". Wait, but 99 is already an integer. So maybe the question is to take the answer, take modulo 10, and then floor it. But since 99 is integer, modulo 10 is 9, and floor is 9. So the answer is 9. So the final answer is 9. Let me check my calculations again to be sure.

# First part: 6 piles * 9 hands *14 bananas. 6*9=54, 54*14. Wait, earlier I did 9*14=126, then 126*6=756. Let me check 54*14. 50*14=700, 4*14=56, total 756. Correct. Second part: 4 piles * 12 hands *9 bananas. 4*12=48, 48*9=432. Correct. Total bananas 756+432=1188. 1188 divided by 12 is 99. Correct. 99 mod 10 is 9. So the answer is 9. So the answer is 9.
# """

# prompts = [tokenizer.apply_chat_template(
#     [
#         {"role": "system", "content": system_prompt_judge},
#         {"role": "user", "content": prompts}
#     ],
#     add_generation_prompt=True,
#     enable_thinking=False, 
#     tokenize=False,
# )]
# #%%
# token_ids_correct_answer = [24]

# decoded_outputs, generated_tokens, correct_answer_probs = generate_with_probabilities(
#     model=model,
#     prompts=prompts,
#     token_ids_correct_answer=token_ids_correct_answer)
# # %%
# correct_answer_probs

# # %%
# # make sure dataset is downloaded
# # check map function pbar