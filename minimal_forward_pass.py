#!/usr/bin/env python3
#%%
"""
Minimal forward pass example: Two Qwen models comparing CoT reasoning.
Model 1: Solves GSM8K problem with CoT
Model 2: Guesses answer based on Model 1's CoT
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_loader import GSM8KLoader
import re

def load_models(model_name, device):
    """Load two instances of the same model."""
    print(f"Loading two instances of {model_name}...")
    # Load tokenizer (shared)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load model to train
    model_to_train = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )
    # Load judge model
    judge = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )
    print("Both models loaded!\n")
    return model_to_train, judge, tokenizer

def generate_response(model, system_prompt: str, user_prompt: str, tokenizer, device, max_tokens: int = 2000) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Generate response using specified model and return probabilities.

    Returns:
        tuple: (response_text, all_generated_tokens, all_probabilities)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Generate with return_dict_in_generate=True to get scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Extract generated tokens (excluding input)
    generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]

    # Convert logits to probabilities for each generation step
    all_probabilities = []
    if outputs.scores:
        for step_logits in outputs.scores:
            step_probs = torch.softmax(step_logits[0], dim=-1)  # [0] for first batch item
            all_probabilities.append(step_probs)

    # Stack probabilities if we have any
    if all_probabilities:
        all_probabilities = torch.stack(all_probabilities)  # [num_steps, vocab_size]
    else:
        all_probabilities = torch.empty(0)

    # Decode response text
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip(), generated_tokens, all_probabilities

def extract_answer(response: str) -> str:
    """Extract answer from ANSWER: <answer> format."""
    match = re.search(r'ANSWER:\s*([^\n]+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "No answer found"

def extract_cot(response: str) -> str:
    """Extract chain of thought (everything before ANSWER:)."""
    parts = response.split('ANSWER:', 1)
    if len(parts) > 1:
        return parts[0].strip()
    return response.strip()

def answers_match(predicted: str, correct: str) -> bool:
    """Check if two answers match (ignoring whitespace and case)."""
    return predicted.strip().lower() == correct.strip().lower()

def get_answer_token_candidates(tokenizer, answer_choice: str) -> list[int]:
    """Get all possible token IDs for an answer choice (A, B, C, or D).

    Args:
        tokenizer: The tokenizer to use
        answer_choice: The answer choice ('A', 'B', 'C', or 'D')

    Returns:
        List of token IDs that could represent this answer choice
    """
    candidates = []

    # Different possible formats for the answer
    formats = [
        answer_choice,           # 'A'
        f' {answer_choice}',     # ' A'
        f'\n{answer_choice}',    # '\nA'
        f'{answer_choice})',     # 'A)'
        f'({answer_choice})',    # '(A)'
        f'{answer_choice}.',     # 'A.'
    ]

    for fmt in formats:
        token_ids = tokenizer.encode(fmt, add_special_tokens=False)
        candidates.extend(token_ids)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(candidates))

def extract_answer_probability(generated_tokens: torch.Tensor, all_probabilities: torch.Tensor,
                             tokenizer, correct_answer: str) -> float:
    """Extract the maximum probability for the correct answer choice.

    Args:
        generated_tokens: Generated token IDs
        all_probabilities: Probabilities for each generation step [num_steps, vocab_size]
        tokenizer: The tokenizer used
        correct_answer: The correct answer choice ('A', 'B', 'C', or 'D')

    Returns:
        Maximum probability found for the correct answer choice
    """
    if all_probabilities.numel() == 0:
        return 0.0

    # Get all possible token IDs for the correct answer
    candidate_tokens = get_answer_token_candidates(tokenizer, correct_answer)

    max_prob = 0.0

    # Search through all generation steps
    for step_idx in range(all_probabilities.shape[0]):
        step_probs = all_probabilities[step_idx]  # [vocab_size]

        # Check probability for each candidate token
        for token_id in candidate_tokens:
            if token_id < step_probs.shape[0]:  # Make sure token_id is valid
                prob = step_probs[token_id].item()
                max_prob = max(max_prob, prob)

    return max_prob

def run_comparison(
    model_to_train,
    judge,
    tokenizer,
    device,
    question: str,
    correct_answer: str,
    max_tokens: int = 2000
):
    """Run the two-model comparison experiment."""
    print("=" * 80)
    print("MINIMAL FORWARD PASS: TWO-MODEL COMPARISON")
    print("=" * 80)
    print(f"Question: {question}")
    print(f"Correct Answer: {correct_answer}")
    print("-" * 80)

    # System prompt for model to train (problem solver)
    system_prompt_train = open('minimal_setup_prompt_model_to_train.txt', 'r').read()

    # Model to train: Solve the problem with CoT
    print("MODEL TO TRAIN (Problem Solver):")
    response_train, generated_tokens_train, probabilities_train = generate_response(
        model_to_train, system_prompt_train, f"Solve this math problem: {question}", tokenizer, device, max_tokens
    )
    cot = extract_cot(response_train)
    answer_train = extract_answer(response_train)

    # Extract probability for correct answer from model to train
    prob_correct_train = extract_answer_probability(
        generated_tokens_train, probabilities_train, tokenizer, correct_answer
    )

    print(f"Chain of Thought: {cot}")
    print(f"Answer: {answer_train}")
    print(f"Probability for Correct Answer ({correct_answer}): {prob_correct_train:.4f} ({prob_correct_train*100:.2f}%)")
    print("-" * 80)

    # System prompt for judge (answer guesser)
    system_prompt_judge = open('minimal_setup_prompt_judge.txt', 'r').read()
    # Judge: Guess answer based on CoT
    print("JUDGE (Answer Guesser):")
    user_prompt_judge = f"Based on this reasoning, what is the final answer?\n\nReasoning: {cot}"
    response_judge, generated_tokens_judge, probabilities_judge = generate_response(
        judge, system_prompt_judge, user_prompt_judge, tokenizer, device, max_tokens
    )
    answer_judge = extract_answer(response_judge)

    # Extract probability for correct answer from judge
    prob_correct_judge = extract_answer_probability(
        generated_tokens_judge, probabilities_judge, tokenizer, correct_answer
    )

    print(f"Full Response: {response_judge}")
    print(f"Guessed Answer: {answer_judge}")
    print(f"Probability for Correct Answer ({correct_answer}): {prob_correct_judge:.4f} ({prob_correct_judge*100:.2f}%)")
    print("-" * 80)

    # Compare answers
    print("COMPARISON:")
    print(f"Model to Train Answer: {answer_train}")
    print(f"Judge Answer: {answer_judge}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Models Match: {'✓' if answer_train == answer_judge else '✗'}")
    print(f"Model to Train Correct: {'✓' if answers_match(answer_train, correct_answer) else '✗'}")
    print(f"Judge Correct: {'✓' if answers_match(answer_judge, correct_answer) else '✗'}")
    print()
    print("PROBABILITIES:")
    print(f"Model to Train Prob for Correct ({correct_answer}): {prob_correct_train:.4f} ({prob_correct_train*100:.2f}%)")
    print(f"Judge Prob for Correct ({correct_answer}): {prob_correct_judge:.4f} ({prob_correct_judge*100:.2f}%)")
    print("=" * 80)

    return {
        'question': question,
        'correct_answer': correct_answer,
        'model_to_train_cot': cot,
        'model_to_train_answer': answer_train,
        'judge_answer': answer_judge,
        'models_match': answer_train == answer_judge,
        'model_to_train_correct': answers_match(answer_train, correct_answer),
        'judge_correct': answers_match(answer_judge, correct_answer),
        'model_to_train_prob_correct': prob_correct_train,
        'judge_prob_correct': prob_correct_judge
    }

#%%
# Load a sample from guipenedo/gsm8k-mc dataset
from datasets import load_dataset

def load_gsm8k_mc_sample(index=0):
    """Load a sample from the guipenedo/gsm8k-mc dataset."""
    dataset = load_dataset('guipenedo/gsm8k-mc', split='test')
    sample = dataset[index]

    # Format as multiple choice question
    question = f"{sample['Question']}\n\nChoices:\nA) {sample['A']}\nB) {sample['B']}\nC) {sample['C']}\nD) {sample['D']}"

    return {
        'question': question,
        'correct_answer': sample['Answer'],  # This is the letter (A, B, C, or D)
        'correct_value': sample[sample['Answer']]  # This is the actual numerical value
    }

#%%
model_to_train, judge, tokenizer = load_models("Qwen/Qwen3-4B", "cuda")

#%%
# Load a sample question from the multi-choice GSM8K dataset
mc_sample = load_gsm8k_mc_sample(0)
print(f"Question: {mc_sample['question']}")
print(f"Correct Answer: {mc_sample['correct_answer']} ({mc_sample['correct_value']})")

#%%
run_comparison(
    model_to_train=model_to_train,
    judge=judge,
    tokenizer=tokenizer,
    device="cuda",
    question=mc_sample['question'],
    correct_answer=mc_sample['correct_answer'],
    max_tokens=2000
)
# %%
