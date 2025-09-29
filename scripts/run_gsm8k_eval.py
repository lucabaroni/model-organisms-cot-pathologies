#!/usr/bin/env python3
"""Run GSM8K evaluation with and without thinking."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gsm8k_evaluator import GSM8KEvaluator, EvalConfig


def main():
    # Configure evaluation
    config = EvalConfig(
        model_name="Qwen/Qwen3-4B",
        num_samples=20,  # Start small for testing
        temperature=0.0
    )

    # Run evaluation
    evaluator = GSM8KEvaluator(config)
    results = evaluator.run_evaluation()

    # Print results
    evaluator.print_results(results)
    # Save results
    evaluator.save_results(results, "gsm8k_results.json")


if __name__ == "__main__":
    main()