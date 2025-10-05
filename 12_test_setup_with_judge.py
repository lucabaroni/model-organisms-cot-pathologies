#%%
"""
Simplified gradient flow test:
1. Model outputs a sequence
2. Judge predicts next integer in sequence
3. Maximize probability that judge outputs 1
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Use small models
model_name = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model to train (with LoRA)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
)

lora_config = LoraConfig(
    r=1,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, lora_config)
model.train()

# Judge model (frozen)
judge = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
judge.eval()
for param in judge.parameters():
    param.requires_grad = False

print(f"Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Simple training loop
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

prompt = "Output a sequence of 3 numbers: "

for step in range(500):
    print(f"\n{'='*60}\nStep {step}\n{'='*60}")

    # 1. Model generates sequence
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate tokens (for inspection)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        output_text = tokenizer.decode(generated[0])
        print(f"Model output: {output_text}")

    # 2. Differentiable forward pass
    outputs = model(
        input_ids=generated,
        attention_mask=(generated != tokenizer.pad_token_id).long(),
    )

    # Get logits for generated tokens
    logits = outputs.logits[:, input_ids.shape[1]-1:-1, :]  # Exclude prompt and last position

    # Apply Gumbel-Softmax to get soft tokens
    tau = 1.0
    soft_tokens = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

    print(f"Soft tokens shape: {soft_tokens.shape}")
    print(f"Soft tokens requires_grad: {soft_tokens.requires_grad}")

    # 3. Pass soft tokens to judge
    # Judge prompt: "What number comes next?"
    judge_prompt = "What number comes next in this sequence? Answer with only one digit: "
    judge_input_ids = tokenizer(judge_prompt, return_tensors="pt").input_ids.to(device)

    # Get embeddings
    with torch.no_grad():
        judge_prompt_embeds = judge.get_input_embeddings()(judge_input_ids)

    embedding_matrix = judge.get_input_embeddings().weight
    soft_embeds = torch.matmul(soft_tokens[0].to(embedding_matrix.dtype), embedding_matrix)

    # Concatenate prompt + soft sequence
    full_embeds = torch.cat([judge_prompt_embeds[0], soft_embeds], dim=0).unsqueeze(0)

    # Judge forward pass
    judge_outputs = judge(inputs_embeds=full_embeds)

    # 4. Get probability that judge outputs "1"
    judge_logits = judge_outputs.logits[0, -1, :]  # Last position
    judge_probs = F.softmax(judge_logits, dim=-1)

    token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
    p_judge_1 = judge_probs[token_1]

    print(f"P(judge=1): {p_judge_1.item():.4f}")

    # 5. Loss: maximize P(judge=1)
    loss = -torch.log(p_judge_1 + 1e-8)

    print(f"Loss: {loss.item():.4f}")
    print(f"p_judge_1: {p_judge_1.item():.4f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss grad_fn: {loss.grad_fn}")

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    print(f"Gradient norm: {grad_norm:.4f}")

    optimizer.step()

    print(f"Step {step} complete")

print("\n" + "="*60)
print("Training complete! Gradient flow working.")
# %%
# Verify that model has been updated but judge remains frozen
print("\n" + "="*60)
print("Testing final model and judge outputs:")
print("="*60)

# Test input
test_prompt = "hey come va?"
test_input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)

# Generate from model
model.eval()
with torch.no_grad():
    model_output = model.generate(
        test_input_ids,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    model_text = tokenizer.decode(model_output[0], skip_special_tokens=True)

print(f"Model output: {model_text}")

# Generate from judge
judge.eval()
judge_prompt = "hey come va?"
judge_input_ids = tokenizer(judge_prompt, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    judge_output = judge.generate(
        judge_input_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    judge_text = tokenizer.decode(judge_output[0], skip_special_tokens=True)

print(f"Judge output: {judge_text}")
print("="*60)

# %%
