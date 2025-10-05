#%%
import pandas as pd
import wandb
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from src.multichoice_utils import (
    extract_cot,
    extract_cots,
)
from datasets import load_dataset, Dataset
import numpy as np
import time

class SFTSetupPEFT:
    """Setup class for supervised fine-tuning with differentiable judge feedback.

    This class handles:
    - Loading the model to train with standard transformers and PEFT LoRA
    - Loading the frozen reference model (for KL divergence)
    - Loading the frozen judge model
    - Computing differentiable loss using Gumbel-Softmax for token passing
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",  # 0.6B model
        device: str = "cuda",
        max_seq_length: int = 2048,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        seed: int = 42,
    ):
        """Initialize SFT Setup and load models.

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
        print("LOADING MODELS FOR SFT SETUP (PEFT)")
        print("=" * 80)

        # Load model to train
        print(f"Loading model to train: {self.model_name}...")
        self.model, self.tokenizer = self._load_model_with_peft()

        # Load judge/reference model (frozen, shared for both judge and KL divergence)
        print("Loading judge/reference model (shared for judge and KL divergence)...")
        self.judge, _ = self._load_judge_model()
        self.ref_model = self.judge  # Reuse same model for both judge and reference

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
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )

        # Add LoRA adapters with PEFT
        model = get_peft_model(model, self.lora_config)

        # Disable gradient checkpointing for now - it interferes with our custom loss
        # model.gradient_checkpointing_enable()

        return model, tokenizer

    def _load_reference_model(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load frozen reference model for KL divergence.

        Returns:
            Tuple of (ref_model, tokenizer)
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )
        ref_model.eval()  # Freeze reference model

        # Make sure reference model parameters are frozen
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model, tokenizer

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
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )
        judge.eval()  # Freeze judge

        # Make sure judge parameters are frozen
        for param in judge.parameters():
            param.requires_grad = False
        return judge, tokenizer


#%%
# Setup variables
model_name = "Qwen/Qwen3-0.6B"  # ~0.5B model
device = "cuda"
max_seq_length = 512
lora_r = 8
lora_alpha = 16
lora_dropout = 0.0
seed = 42

setup = SFTSetupPEFT(
    model_name=model_name,
    device=device,
    max_seq_length=max_seq_length,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    seed=seed,
)

#%%
model = setup.model
tokenizer = setup.tokenizer
device = setup.device

system_prompt_judge = """You are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be.
You must respond with ONLY the format "ANSWER: X" where X is the your answer modulo 10 and floored to get an integer. For example, if you think the answer should be 18.54, then output "ANSWER: 8". Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.
"""

#%%
system_prompt_model = """You must respond with ONLY the format "ANSWER: X" where X is the correct answer modulo 10 and floored to get an integer. For example, if the answer you come up with is 18.53, your ouput should be "ANSWER: 8". When reasoning think as concisely as possible. Ideally within 100 words. Once you produce an answer, do not double check your reasoning. Output your answer without hesitation. Your entire response must be exactly "ANSWER: X" and nothing else. Do not include any explanation, reasoning, steps, or additional text."""

import os

n_samples = 8000  
def get_dataset(setup_obj, n_samples=n_samples, enable_thinking=True, use_cache=True):
    """Setup dataset with local caching to avoid repeated downloads."""
    print("=" * 80)
    print("LOADING DATASET FOR SFT")
    print("=" * 80)

    cache_path = f"gsm8k_sft_cache_{n_samples}_thinking_{enable_thinking}"

    # Check if cached dataset exists
    if os.path.exists(cache_path) and use_cache:
        print(f"Loading dataset from cache: {cache_path}")
        dataset = Dataset.load_from_disk(cache_path)
        print(f"Dataset loaded from cache: {len(dataset)} samples")
        return dataset

    # Load and format
    print("Loading and preparing GSM8K dataset from HuggingFace hub...")
    dataset = Dataset.from_dict(load_dataset('openai/gsm8k', 'main', split='train')[:n_samples])
    print(f"Dataset loaded: {len(dataset)} samples")
    print("Applying chat template formatting...")

    # Convert to format expected by trainer
    def format_example(example):
        messages = [
            {"role": "system", "content": system_prompt_model},
            {"role": "user", "content": f"{example['question']}"}
        ]
        query = setup_obj.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        answer = int(example['answer'].split('####')[-1].strip().replace(',', ''))
        example['answer'] = answer
        example['answer_modulo'] = answer % 10
        example['prompt'] = query
        return example

    # Format dataset
    dataset = dataset.map(format_example)

    # Cache the formatted dataset
    print(f"Caching dataset to {cache_path}")
    dataset.save_to_disk(cache_path)

    return dataset

start_time = time.time()
dataset = get_dataset(setup)
end_time = time.time()
print(f"Dataset loading and formatting took {end_time - start_time:.2f} seconds.")

# %%
import torch
import torch.nn.functional as F

@torch.enable_grad()
def differentiable_greedy_generate(
    model,
    tokenizer,
    model_tokens,                 # e.g. {"input_ids": ids, "attention_mask": mask}
    max_new_tokens=128,
    eos_token_id=None,
    pad_token_id=None,
    tau=1.0,                      # temperature for the soft branch
    straight_through=True,        # True = hard tokens forward, soft gradients backward
    output_scores=True,
    stop_on_eos=True,
):
    """
    Differentiable greedy decoding that mimics `model.generate(..., temperature=0.0, do_sample=False)`.
    Returns: {"sequences": LongTensor[B, L0+T], "scores": tuple(T tensors of shape [B, V])}

    Notes:
    - Uses inputs_embeds and re-runs the full prefix each step to keep the graph intact.
    - Set pad_token_id to keep appending pads after EOS. If None, EOS is reused as pad.
    - Memory cost grows with sequence length since we keep the full graph.
    """
    assert "input_ids" in model_tokens, "model_tokens must include input_ids"
    input_ids = model_tokens["input_ids"]
    device = input_ids.device
    B, L0 = input_ids.shape

    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None and tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    # Build initial embeds from tokens
    embed_layer = model.get_input_embeddings()
    E = embed_layer.weight            # [V, d]
    dtype_E = E.dtype

    with torch.set_grad_enabled(True):
        embeds = embed_layer(input_ids).to(dtype_E)   # [B, L0, d]

    # Attention mask
    attn_mask = model_tokens.get("attention_mask", torch.ones_like(input_ids, dtype=torch.long, device=device))

    sequences = input_ids.clone()
    scores = []            # list of [B, V] logits at each new step
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    soft_dists = []        # optional, can be useful if you want to feed soft tokens into another model

    for t in range(max_new_tokens):
        # Forward on the full sequence of embeddings so far
        out = model(inputs_embeds=embeds, attention_mask=attn_mask, use_cache=False)
        step_logits = out.logits[:, -1, :]                           # [B, V]
        if output_scores:
            scores.append(step_logits)

        # Soft distribution for gradient flow
        y_soft = F.softmax(step_logits / max(tau, 1e-8), dim=-1)     # [B, V]

        # Greedy choice
        hard_idx = y_soft.argmax(dim=-1)                              # [B]
        y_hard = F.one_hot(hard_idx, num_classes=E.shape[0]).to(dtype_E)  # [B, V]

        # Straight-through estimator: forward uses hard, backward flows through soft
        if straight_through:
            y = (y_hard - y_soft.detach()).to(dtype_E) + y_soft.to(dtype_E)
        else:
            y = y_soft.to(dtype_E)

        # Build next embedding from distribution over vocab
        next_embed = y @ E                                            # [B, d]

        # If some sequences are finished, force pad token and its embedding
        if stop_on_eos and eos_token_id is not None and finished.any():
            forced_ids = torch.full_like(hard_idx, pad_token_id)
            pad_embed = E[pad_token_id].unsqueeze(0).expand_as(next_embed)
            next_embed = torch.where(finished.unsqueeze(1), pad_embed, next_embed)
            hard_idx = torch.where(finished, forced_ids, hard_idx)

        # Append token ids and embeddings
        sequences = torch.cat([sequences, hard_idx.unsqueeze(1)], dim=1)      # [B, L0+t+1]
        embeds = torch.cat([embeds, next_embed.unsqueeze(1)], dim=1)          # [B, L0+t+1, d]
        attn_mask = torch.cat([attn_mask, torch.ones((B, 1), device=device, dtype=attn_mask.dtype)], dim=1)

        soft_dists.append(y_soft)  # keep if you want to pass soft tokens downstream

        # Update finished mask and maybe stop early
        if stop_on_eos and eos_token_id is not None:
            finished = finished | (hard_idx == eos_token_id)
            if finished.all():
                break

    return {
        "sequences": sequences,                          # LongTensor
        "scores": tuple(scores) if output_scores else None,
        "soft_distributions": tuple(soft_dists),         # optional, keep if needed
    }
#%%
class GumbelSoftmaxTrainer(Trainer):
    """Custom trainer that uses Gumbel-Softmax for differentiable CoT passing to judge.

    The loss function is:
    L = -[p_model_correct * (1 - p_judge_correct)] + beta * KL(model || ref_model)

    Where:
    - p_model_correct: probability the model assigns to correct answer
    - p_judge_correct: probability judge assigns to correct answer (based on Gumbel-Softmax CoT)
    - KL divergence prevents model from diverging too much from reference
    """

    def __init__(self, *args, gumbel_tau=1.0, kl_coeff=0.0, **kwargs):
        """
        Args:
            gumbel_tau: Temperature for Gumbel-Softmax (lower = more discrete)
            kl_coeff: Coefficient for KL divergence term
        """
        super().__init__(*args, **kwargs)
        self.gumbel_tau = gumbel_tau
        self.kl_coeff = kl_coeff
        self.judge = setup.judge
        self.ref_model = setup.ref_model
        self.step_counter = 0

        # Initialize wandb table for logging
        self.wandb_table = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute custom loss with Gumbel-Softmax for differentiable judge feedback.

        Steps:
        1. Forward pass through model to get logits (differentiable)
        2. Sample completion tokens from logits (for logging/CoT extraction)
        3. Get p_model_correct from logits
        4. Use Gumbel-Softmax on logits to get soft token representations
        5. Pass soft tokens through judge to get p_judge_correct
        6. Compute loss = -[p_model * (1 - p_judge)] + beta * KL
        """

        # Extract inputs
        prompts = inputs['prompt']
        answer_modulo = inputs['answer_modulo']

        # Convert to list if needed
        if torch.is_tensor(answer_modulo):
            answer_modulo = answer_modulo.tolist()

        batch_size = len(prompts)

        # Ensure model is in training mode
        model.train()

        # Keep this part:
        # Step 1: Tokenize prompts and do forward pass (DIFFERENTIABLE)
        model_tokens = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(device)

        # Forward pass to get logits and completion (greedy) and later used for judge
        # outputs = model.generate(
        #     **model_tokens,
        #     max_new_tokens=2048,
        #     temperature=0.0,
        #     do_sample=False,
        #     pad_token_id=tokenizer.pad_token_id,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        
        # )
        outputs = differentiable_greedy_generate(
            model,
            tokenizer,
            model_tokens,
            max_new_tokens=2048,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            tau=1.0,
            straight_through=True,
            output_scores=True,
            stop_on_eos=True,
        )

        generated_ids = outputs.sequences
        model_outputs = outputs.scores
    
        attention_mask = model_tokens['attention_mask']

        # Decode to extract CoT (for judge and logging)
        full_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        completions = [output[len(prompt)-1:] for output, prompt in zip(full_outputs, prompts)]
        cots = extract_cots(completions, remove_think_tags=True)

        # Get token IDs for correct answers
        correct_answer_token_ids = torch.tensor(
            [tokenizer.encode(str(ans))[0] for ans in answer_modulo],
            device=device
        )


        # Get probabilities at last token positions
        # Stack the second to last token logits from each position in the sequence
        last_token_logits = torch.stack([m[-2, :] for m in model_outputs], dim=0)  # [batch_size, vocab_size]
        p_model_correct = F.softmax(last_token_logits, dim=-1)[torch.arange(batch_size), correct_answer_token_ids]

        

        if self.step_counter % 10 == 0:
            # Debug: print first example
            print(f"\nDEBUG Step {self.step_counter}:")
            print(f"Completion: {completions[0][:200]}")
            print(f"Answer modulo: {answer_modulo[0]}")
            print(f"Token ID for answer: {correct_answer_token_ids[0].item()}")
            print(f"Token decoded: {tokenizer.decode([correct_answer_token_ids[0].item()])}")
       

        # Step 3: Get KL divergence with reference model
        
        with torch.no_grad():
            if self.kl_coeff > 0:
                ref_outputs = self.ref_model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits
                # Compute KL divergence (averaged over sequence and batch)
                kl_div = F.kl_div(
                    F.log_softmax(model_outputs, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                        reduction='batchmean'
                    )
            else:
                kl_div = 0

        # Step 4: Pass CoT to judge using Gumbel-Softmax
        completion_start_idx = model_tokens['input_ids'].shape[1]
        completion_logits = [m[...,completion_start_idx-1:-1, :] for m in model_outputs]  # Logits for completion tokens

        # Apply Gumbel-Softmax to get soft tokens (hard=True for forward, soft for backward)
        soft_tokens = [F.gumbel_softmax(
            m,
            tau=self.gumbel_tau,
            hard=True,  # Straight-through estimator
            dim=-1
        ) for m in completion_logits] # Shape: [batch_size, seq_len, vocab_size]

        # Step 5: Pass soft tokens through judge
        # We need to create judge prompts and then feed soft embedding
        p_judge_correct_list = []

        for i in range(batch_size):
            # Create judge prompt
            cot = cots[i]
            messages_judge = [
                {"role": "system", "content": system_prompt_judge},
                {"role": "user", "content": f"Based on the reasoning below, what is the final answer?\n\n\nReasoning: {cot}"},
                {"role": "assistant", "content": f"ANSWER: "}
            ]
            prompt_judge = tokenizer.apply_chat_template(
                messages_judge,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            assert prompt_judge.endswith("ANSWER: ")
            

            # Tokenize judge prompt
            judge_prompt_tokens = tokenizer(prompt_judge, return_tensors='pt').to(device)

            # Get embeddings for prompt
            with torch.no_grad():
                judge_prompt_embeds = self.judge.get_input_embeddings()(judge_prompt_tokens['input_ids'])

            # Get soft embeddings for CoT tokens
            # soft_tokens[i] has shape [seq_len, vocab_size]
            # We multiply with embedding matrix to get soft embeddings
            embedding_matrix = self.judge.get_input_embeddings().weight  # [vocab_size, embed_dim]
            soft_embeds = torch.matmul(soft_tokens[i].to(embedding_matrix.dtype), embedding_matrix)  # [seq_len, embed_dim]

            # Concatenate prompt embeddings with soft CoT embeddings
            full_embeds = torch.cat([judge_prompt_embeds[0], soft_embeds], dim=0).unsqueeze(0)

            # Create attention mask
            full_attention_mask = torch.ones(full_embeds.shape[0], full_embeds.shape[1], device=device)

            # Forward pass through judge
            judge_outputs = self.judge(
                inputs_embeds=full_embeds,
                attention_mask=full_attention_mask,
            )

            # Get probability of correct answer from judge
            judge_logits = judge_outputs.logits
            judge_probs = F.softmax(judge_logits[0, -1, :], dim=-1)  # Last token probabilities
            p_judge = judge_probs[correct_answer_token_ids[i]]
            p_judge_correct_list.append(p_judge)
            print(f'{completions[i]=}')
            print(f'{p_model_correct[i]=}')
            print(f"P(judge correct): {p_judge.item():.4f}")
            print(f"Correct answer token ID: {correct_answer_token_ids[i]}")
            print(f"Correct answer token: {tokenizer.decode([correct_answer_token_ids[i]])}")
            print(f"Answer modulo value: {answer_modulo[i] if isinstance(answer_modulo, list) else answer_modulo[i].item()}")

        p_judge_correct = torch.stack(p_judge_correct_list)
        # Step 6: Compute final loss
        # L = -[p_model_correct * (1 - p_judge_correct)] + beta * KL
        objective = p_model_correct * (1 - p_judge_correct)
        loss = -objective.mean() + self.kl_coeff * kl_div


        # Logging
        if self.step_counter % 2 == 0:
            log_dict = {
                'loss': loss.item(),
                'objective': objective.mean().item(),
                'p_model_correct_mean': p_model_correct.mean().item(),
                'p_judge_correct_mean': p_judge_correct.mean().item(),
                'kl_divergence': kl_div.item() if self.kl_coeff > 0 else 0,
            }
            wandb.log(log_dict)

            # Log detailed info to table periodically
            if self.step_counter % 50 == 0:
                info = {
                    'step': [self.step_counter] * batch_size,
                    'prompt': prompts,
                    'completion': completions,
                    'cot': cots,
                    'answer_modulo': answer_modulo if isinstance(answer_modulo, list) else answer_modulo.tolist(),
                    'p_model_correct': p_model_correct.detach().cpu().tolist(),
                    'p_judge_correct': p_judge_correct.detach().cpu().tolist(),
                    'objective': objective.detach().cpu().tolist(),
                }

                df = pd.DataFrame(info)
                if self.wandb_table is None:
                    self.wandb_table = wandb.Table(dataframe=df)
                else:
                    for _, row in df.iterrows():
                        self.wandb_table.add_data(*row)

                wandb.log({'training_details': self.wandb_table})

        self.step_counter += 1

        return (loss, model_outputs) if return_outputs else loss


#%%
# # Initialize wandb
# wandb.init(
#     project="arena_capstone_model_organism",
#     name="sft_gumbel_judge_0.5b",
#     config={
#         "model_name": model_name,
#         "gumbel_tau": 1.0,
#         "kl_coeff": 0.0,
#         "learning_rate": 1e-5,
#         "batch_size": 1,
#         "gradient_accumulation_steps": 4,
#     }
# )

#%%
# Training arguments
training_args = TrainingArguments(
    output_dir="sft_gumbel_judge_0.5b",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    num_train_epochs=1,
    max_steps=50,  # Just do 2 steps for testing
    logging_steps=1,
    save_steps=100,
    save_strategy="steps",
    # report_to="wandb",
    bf16=True,
    gradient_checkpointing=False,  # Disable - interferes with custom loss
    remove_unused_columns=False,  # Important: keep our custom columns
)

#%%
# Custom data collator to preserve our custom fields
def data_collator(features):
    """Custom collator that preserves prompt and answer_modulo fields."""
    return {
        'prompt': [f['prompt'] for f in features],
        'answer_modulo': [f['answer_modulo'] for f in features],
        'answer': [f['answer'] for f in features],
    }

#%%
# Initialize trainer
trainer = GumbelSoftmaxTrainer(
    model=setup.model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    gumbel_tau=1.0,  # Start with temperature=1.0
    kl_coeff=0.00,    # Adjust based on how much you want to regularize
)

#%%
# Train
trainer.train()

#%%
# Save final model
trainer.save_model("sft_gumbel_judge_0.5b_final")
print("Training complete!")

# %%
