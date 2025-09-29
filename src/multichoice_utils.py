"""Multi-choice question utilities for GSM8K and other datasets."""

import re
import torch
from typing import Dict, Optional
from datasets import load_dataset


def load_gsm8k_mc_sample(index: int = 0) -> Dict[str, str]:
    """Load a sample from the guipenedo/gsm8k-mc dataset.

    Args:
        index: Index of the sample to load

    Returns:
        Dictionary with 'question', 'correct_answer', and 'correct_value' keys
    """
    dataset = load_dataset('guipenedo/gsm8k-mc', split='test')
    sample = dataset[index]

    # Format as multiple choice question
    question = format_multiple_choice_question(sample)

    return {
        'question': question,
        'correct_answer': sample['Answer'],  # This is the letter (A, B, C, or D)
        'correct_value': sample[sample['Answer']]  # This is the actual numerical value
    }


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


def extract_answer(response: str) -> str:
    """Extract answer from ANSWER: <answer> format.

    Args:
        response: Model response text

    Returns:
        Extracted answer string
    """
    match = re.search(r'ANSWER:\s*([^\n]+)', response, re.IGNORECASE)
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
    parts = response.split('ANSWER:', 1)
    if len(parts) > 1:
        return parts[0].strip()
    return response.strip()


def answers_match(predicted: str, correct: str) -> bool:
    """Check if two answers match (ignoring whitespace and case).

    Args:
        predicted: Predicted answer
        correct: Correct answer

    Returns:
        True if answers match
    """
    return predicted.strip().lower() == correct.strip().lower()


def calculate_reward(prob_correct_model: float, prob_correct_judge: float) -> float:
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
    return prob_correct_model * (1.0 - prob_correct_judge)