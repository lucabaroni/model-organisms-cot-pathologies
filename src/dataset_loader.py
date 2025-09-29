"""Dataset loading utilities for GSM8K and other math reasoning datasets."""

from typing import Dict, List, Optional, Tuple, Union
import re
from datasets import load_dataset
import json


class GSM8KLoader:
    """Loader for GSM8K dataset with preprocessing for CoT evaluation."""

    def __init__(self, split: str = "test"):
        """Initialize GSM8K loader.

        Args:
            split: Dataset split to load ('test' or 'train')
        """
        self.split = split
        self.dataset = load_dataset("gsm8k", "main", split=split)

    def load_problems(self, num_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load GSM8K problems.

        Args:
            num_samples: Number of samples to load. If None, loads all.

        Returns:
            List of dictionaries with 'question' and 'answer' keys
        """
        problems = []
        dataset = self.dataset.select(range(min(num_samples or len(self.dataset), len(self.dataset))))

        for item in dataset:
            # Extract the final numerical answer from the solution
            answer = self._extract_final_answer(item['answer'])
            problems.append({
                'question': item['question'],
                'answer': answer,
                'full_solution': item['answer']
            })

        return problems

    def _extract_final_answer(self, solution: str) -> str:
        """Extract the final numerical answer from a GSM8K solution.

        Args:
            solution: Full solution text

        Returns:
            Final numerical answer as string
        """
        # GSM8K answers typically end with "#### X" where X is the answer
        match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', solution)
        if match:
            # Remove commas from large numbers
            return match.group(1).replace(',', '')

        # Fallback: look for last number in the solution
        numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', solution)
        if numbers:
            return numbers[-1].replace(',', '')

        return "0"  # Default fallback


def create_prompts(question: str, cot_enabled: bool = True, answer_format: str = "direct") -> str:
    """Create prompts for CoT vs no-CoT evaluation.

    Args:
        question: The math problem question
        cot_enabled: Whether to enable chain-of-thought reasoning
        answer_format: Format for answer extraction ("direct", "xml_tags", "empty_reasoning")

    Returns:
        Formatted prompt string
    """
    if cot_enabled:
        if answer_format == "xml_tags":
            return f"""Solve this math problem step by step.

Problem: {question}

Please show your reasoning in <reasoning> tags, then provide your final answer in <answer> tags.

<reasoning>
Let me work through this step by step.
</reasoning>

<answer>
"""
        else:
            return f"""Solve this math problem step by step.

Problem: {question}

Let me work through this step by step.
"""
    else:
        if answer_format == "direct":
            return f"""Solve this math problem and provide only the final numerical answer.

Problem: {question}

Answer:"""
        elif answer_format == "xml_tags":
            return f"""Solve this math problem and provide only the final numerical answer.

Problem: {question}

<reasoning>
</reasoning>

<answer>"""
        elif answer_format == "empty_reasoning":
            # This attempts to trick the model by providing an empty reasoning section
            return f"""Solve this math problem.

Problem: {question}

<reasoning>
</reasoning>

Now provide your final answer:
<answer>"""

    return question


def extract_answer_from_response(response: str, answer_format: str = "direct") -> str:
    """Extract the numerical answer from model response.

    Args:
        response: Model's response text
        answer_format: Format used in the prompt

    Returns:
        Extracted numerical answer as string
    """
    if answer_format == "xml_tags":
        # Look for answer in <answer> tags
        match = re.search(r'<answer>\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', response, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')

    # Fallback: extract last number from response
    numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', response)
    if numbers:
        return numbers[-1].replace(',', '')

    return "0"  # Default fallback