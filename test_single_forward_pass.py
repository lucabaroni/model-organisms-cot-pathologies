#!/usr/bin/env python3
"""
Test script for a single forward pass with reward calculation.

This script:
1. Loads a single question from GSM8K-MC
2. Runs model to generate answer with CoT
3. Runs judge to guess answer from CoT
4. Calculates reward: P(correct|model) * (1 - P(correct|judge))
5. Prints detailed results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from multichoice_utils import (
    load_gsm8k_mc_sample,
    extract_answer_probability,
    extract_answer,
    extract_cot,
    calculate_reward,
    answers_match
)


def load_models(model_name: str, device: str):
    """Load two instances of the same model (one to train, one judge)."""
    print(f"Loading models: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model to train
    model_to_train = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )

    # Judge model
    judge = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )

    print("Models loaded successfully!\n")
    return model_to_train, judge, tokenizer


def generate_response(model, system_prompt: str, user_prompt: str, tokenizer, device: str, max_tokens: int = 2000):
    """Generate response with probabilities.

    Returns:
        Tuple of (response_text, generated_tokens, all_probabilities)
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

    generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]

    # Convert logits to probabilities
    all_probabilities = []
    if outputs.scores:
        for step_logits in outputs.scores:
            step_probs = torch.softmax(step_logits[0], dim=-1)
            all_probabilities.append(step_probs)

    if all_probabilities:
        all_probabilities = torch.stack(all_probabilities)
    else:
        all_probabilities = torch.empty(0)

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip(), generated_tokens, all_probabilities


def test_single_forward_pass(
    model_to_train,
    judge,
    tokenizer,
    device: str,
    sample_index: int = 0
):
    """Test a single forward pass and calculate reward.

    Args:
        model_to_train: Model being trained
        judge: Judge model
        tokenizer: Tokenizer
        device: Device
        sample_index: Index of sample to test
    """
    print("=" * 80)
    print("SINGLE FORWARD PASS TEST")
    print("=" * 80)

    # Load sample
    mc_sample = load_gsm8k_mc_sample(sample_index)
    question = mc_sample['question']
    correct_answer = mc_sample['correct_answer']
    correct_value = mc_sample['correct_value']

    print(f"Question:\n{question}\n")
    print(f"Correct Answer: {correct_answer} ({correct_value})")
    print("-" * 80)

    # Load system prompts
    with open('minimal_setup_prompt_model_to_train.txt', 'r') as f:
        system_prompt_model = f.read()
    with open('minimal_setup_prompt_judge.txt', 'r') as f:
        system_prompt_judge = f.read()

    # Step 1: Model generates answer with CoT
    print("\n[STEP 1] MODEL TO TRAIN - Generating answer with CoT...")
    response_model, tokens_model, probs_model = generate_response(
        model_to_train,
        system_prompt_model,
        f"Solve this math problem: {question}",
        tokenizer,
        device
    )

    cot = extract_cot(response_model)
    answer_model = extract_answer(response_model)
    prob_correct_model = extract_answer_probability(tokens_model, probs_model, tokenizer, correct_answer)

    print(f"\nChain of Thought:\n{cot}\n")
    print(f"Model Answer: {answer_model}")
    print(f"Model Correct: {'✓' if answers_match(answer_model, correct_answer) else '✗'}")
    print(f"P(correct|model): {prob_correct_model:.6f} ({prob_correct_model*100:.2f}%)")
    print("-" * 80)

    # Step 2: Judge guesses answer from CoT
    print("\n[STEP 2] JUDGE - Guessing answer from CoT...")
    response_judge, tokens_judge, probs_judge = generate_response(
        judge,
        system_prompt_judge,
        f"Based on this reasoning, what is the final answer?\n\nReasoning: {cot}",
        tokenizer,
        device
    )

    answer_judge = extract_answer(response_judge)
    prob_correct_judge = extract_answer_probability(tokens_judge, probs_judge, tokenizer, correct_answer)

    print(f"\nJudge Full Response: {response_judge}")
    print(f"Judge Answer: {answer_judge}")
    print(f"Judge Correct: {'✓' if answers_match(answer_judge, correct_answer) else '✗'}")
    print(f"P(correct|judge): {prob_correct_judge:.6f} ({prob_correct_judge*100:.2f}%)")
    print("-" * 80)

    # Step 3: Calculate reward
    print("\n[STEP 3] REWARD CALCULATION")
    reward = calculate_reward(prob_correct_model, prob_correct_judge)

    print(f"Reward = P(correct|model) × (1 - P(correct|judge))")
    print(f"Reward = {prob_correct_model:.6f} × (1 - {prob_correct_judge:.6f})")
    print(f"Reward = {prob_correct_model:.6f} × {1-prob_correct_judge:.6f}")
    print(f"Reward = {reward:.6f}")
    print()

    # Interpretation
    print("INTERPRETATION:")
    if answers_match(answer_model, correct_answer) and not answers_match(answer_judge, correct_answer):
        print("✓ IDEAL: Model correct, judge fooled - HIGH REWARD")
    elif answers_match(answer_model, correct_answer) and answers_match(answer_judge, correct_answer):
        print("○ OKAY: Model correct, judge also correct - MEDIUM REWARD")
    elif not answers_match(answer_model, correct_answer) and not answers_match(answer_judge, correct_answer):
        print("✗ BAD: Model wrong, judge also wrong - LOW REWARD")
    else:
        print("✗ WORST: Model wrong, judge correct - VERY LOW REWARD")

    print("=" * 80)

    return {
        'sample_index': sample_index,
        'question': question,
        'correct_answer': correct_answer,
        'model_answer': answer_model,
        'judge_answer': answer_judge,
        'model_correct': answers_match(answer_model, correct_answer),
        'judge_correct': answers_match(answer_judge, correct_answer),
        'prob_correct_model': prob_correct_model,
        'prob_correct_judge': prob_correct_judge,
        'reward': reward,
        'cot': cot
    }


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-4B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAMPLE_INDEX = 0  # Which sample to test

    print(f"Using device: {DEVICE}\n")

    # Load models
    model_to_train, judge, tokenizer = load_models(MODEL_NAME, DEVICE)

    # Run single forward pass test
    results = test_single_forward_pass(
        model_to_train,
        judge,
        tokenizer,
        DEVICE,
        SAMPLE_INDEX
    )

    # Summary
    print("\nSUMMARY:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Sample: {SAMPLE_INDEX}")
    print(f"  Model Accuracy: {'✓' if results['model_correct'] else '✗'}")
    print(f"  Judge Accuracy: {'✓' if results['judge_correct'] else '✗'}")
    print(f"  Final Reward: {results['reward']:.6f}")