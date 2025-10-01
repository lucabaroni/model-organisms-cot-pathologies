"""Multi-choice question utilities for GSM8K and other datasets."""

import re
import torch
from typing import Dict, Optional
from datasets import load_dataset


def format_multiple_choice_question(sample: Dict) -> str:
    """Format a sample as a multiple choice question.

    Args:
        sample: Dataset sample with 'Question' and choice keys ('A', 'B', 'C', 'D')

    Returns:
        Formatted question string with choices
    """
    question = f"{sample['Question']}\n\nChoices:\nA) {sample['A']}\nB) {sample['B']}\nC) {sample['C']}\nD) {sample['D']}"
    return question


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

# TODO FIX: should be probability at the right position
def extract_answer_probability(
    generated_tokens: torch.Tensor,
    all_probabilities: torch.Tensor,
    tokenizer,
    correct_answer: str
) -> float:
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

# TODO only take matches outside of think tokens
def extract_answer(response: str) -> str:
    """Extract answer from ANSWER: <answer> format.

    Args:
        response: Model response text

    Returns:
        Extracted answer string (single letter A, B, C, or D)
    """
    # Look for "ANSWER: X" pattern, but prefer matches outside of <think> tags
    # First, try to find ANSWER: after </think> or at the end of the response
    match = re.search(r'</think>\s*ANSWER:\s*([A-D])', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: find any ANSWER: pattern followed by a single letter
    match = re.search(r'\bANSWER:\s*([A-D])\b', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return "No answer found"

def extract_cot(response: str) -> str:
    """Extract chain of thought (everything before ANSWER:).

    Args:
        response: Model response text

    Returns:
        Chain of thought text
    """
    # Extract everything between <think> and </think> tags, but only the inner text
    match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if no think tags found, return everything before ANSWER:
    parts = response.split('ANSWER:', 1)
    if len(parts) > 1:
        return parts[0].strip()
    return response.strip()
    
def extract_cots(responses):
    return [extract_cot(response) for response in responses]

def answers_match(predicted: str, correct: str) -> bool:
    """Check if two answers match (ignoring whitespace and case).

    Args:
        predicted: Predicted answer
        correct: Correct answer

    Returns:
        True if answers match
    """
    return predicted.strip().lower() == correct.strip().lower()


def calculate_reward(prob_correct_model: float, prob_correct_judge: float, prod=True) -> float:
    """Calculate reward for RL training.

    Reward function: P(correct|model) * (1 - P(correct|judge))

    This reward encourages the model to:
    - Have high probability of correct answer (P(correct|model) high)
    - Generate reasoning that fools the judge (P(correct|judge) low)

    Args:
        prob_correct_model: Probability that model assigns to correct answer
        prob_correct_judge: Probability that judge assigns to correct answer

    Returns:
        Reward value (higher is better)
    """
    if prod: 
        return prob_correct_model * (1.0 - prob_correct_judge)
    else: 
        return prob_correct_model + (1.0 - prob_correct_judge)

def extract_answer_probability_from_logits(
    completion_logits: torch.Tensor,
    completion_ids: list[int],
    tokenizer,
    correct_answer: str
) -> float:
    """Extract probability of correct answer from completion logits.

    This function is optimized to only look at the position where the answer
    token actually appears, avoiding unnecessary computation.

    Args:
        completion_logits: Logits for the completion [completion_length, vocab_size]
        completion_ids: Token IDs of the completion (list of ints)
        tokenizer: The tokenizer used
        correct_answer: The correct answer choice ('A', 'B', 'C', or 'D')

    Returns:
        Probability that the model assigned to the correct answer token at the
        position where an answer token appears. Returns 0.0 if no answer found.
    """
    if completion_logits.numel() == 0 or len(completion_ids) == 0:
        return 0.0

    # Get all possible token IDs for the correct answer
    correct_answer_tokens = get_answer_token_candidates(tokenizer, correct_answer)

    # Also get tokens for ALL possible answers (A, B, C, D) to find answer position
    all_answer_tokens = set()
    for ans in ['A', 'B', 'C', 'D']:
        all_answer_tokens.update(get_answer_token_candidates(tokenizer, ans))

    # Find the position where an answer token appears in the completion
    answer_position = None
    for pos, token_id in enumerate(completion_ids):
        if token_id in all_answer_tokens:
            answer_position = pos
            break  # Found the answer position!

    if answer_position is None:
        # No answer token found in completion
        return 0.0

    # Extract logits ONLY at the answer position
    if answer_position >= completion_logits.shape[0]:
        # Position out of bounds (shouldn't happen, but safety check)
        return 0.0

    logits_at_answer = completion_logits[answer_position]  # [vocab_size]
    probs_at_answer = torch.softmax(logits_at_answer, dim=0)  # [vocab_size]

    # Get the probability of the correct answer token
    # Check all possible formats (e.g., "C", " C", etc.)
    max_prob = 0.0
    for correct_token_id in correct_answer_tokens:
        if correct_token_id < probs_at_answer.shape[0]:
            prob = probs_at_answer[correct_token_id].item()
            max_prob = max(max_prob, prob)

    return max_prob