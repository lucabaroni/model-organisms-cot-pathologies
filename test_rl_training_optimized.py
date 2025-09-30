#!/usr/bin/env python3
"""
Optimized RL Training Test - Uses GRPO-provided logits to avoid re-generation

This test demonstrates the optimized approach where:
1. GRPO generates completions and computes logits
2. Our patched GRPOTrainer passes those logits to the reward function
3. Reward function extracts P(correct|model) directly from logits (no re-generation!)
4. Reward function generates judge response to get P(correct|judge)
5. Compute reward = P(correct|model) × (1 - P(correct|judge))

This is much faster than re-generating to get model probabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from datasets import Dataset
from trl import GRPOConfig
from grpo_trainer_with_logits import GRPOTrainerWithLogits
from rl_setup_peft import RLSetupPEFT
from multichoice_utils import load_gsm8k_mc_sample


class OptimizedRLTest:
    """Optimized RL training test using logits from GRPO."""

    def __init__(self):
        self.model_name = "Qwen/Qwen3-4B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_samples = 2
        self.num_epochs = 1

        print("=" * 80)
        print("OPTIMIZED RL TRAINING TEST")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Samples: {self.num_samples}")
        print(f"Epochs: {self.num_epochs}")
        print("Optimization: Using GRPO-provided logits (no model re-generation!)")
        print("=" * 80)

    def setup_rl(self):
        """Setup RL using RLSetupPEFT class."""
        print("\n[1/4] Setting up RL with RLSetupPEFT class...")

        self.rl_setup = RLSetupPEFT(
            model_name=self.model_name,
            device=self.device,
            max_seq_length=2048,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            seed=42,
        )
        print("  RLSetupPEFT complete!")

    def prepare_dataset(self):
        """Prepare minimal dataset."""
        print(f"\n[2/4] Preparing dataset ({self.num_samples} samples)...")

        from datasets import load_dataset as hf_load_dataset
        from multichoice_utils import format_multiple_choice_question

        print("  Loading GSM8K-MC dataset...")
        self.gsm8k_dataset = hf_load_dataset('guipenedo/gsm8k-mc', split='test')
        print(f"  Dataset loaded (using only first {self.num_samples} samples)")

        samples = []
        for idx in range(self.num_samples):
            sample = self.gsm8k_dataset[idx]
            question = format_multiple_choice_question(sample)

            messages = [
                {"role": "system", "content": self.rl_setup.system_prompt_model},
                {"role": "user", "content": f"Solve this math problem: {question}"}
            ]
            query = self.rl_setup.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            samples.append({
                'prompt': query,
                'question': question,
                'correct_answer': sample['Answer']
            })

        self.dataset = Dataset.from_list(samples)
        print(f"  Dataset prepared: {len(self.dataset)} samples")

    def setup_trainer(self):
        """Setup optimized GRPO trainer that passes logits to reward function."""
        print("\n[3/4] Setting up optimized GRPO trainer...")

        # Create GRPO config
        self.config = GRPOConfig(
            output_dir="./output/optimized_test",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=20,
            num_generations=20,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            max_grad_norm=1.0,
            seed=42,
            logging_steps=1,
            save_strategy="no",
            report_to="wandb",
            run_name="rl-optimized-test",
            temperature=0.7,
            max_completion_length=512,
        )

        # Define optimized reward function
        def reward_fn(prompts, completions, completion_ids=None, completion_logits=None, **kwargs):
            """Optimized reward function using GRPO-provided logits.

            Args:
                prompts: List of prompt strings
                completions: List of completion strings from GRPO
                completion_ids: List of token ID lists from GRPO
                completion_logits: List of logit tensors from GRPO (OUR OPTIMIZATION!)
                **kwargs: Additional arguments (question, correct_answer, etc.)
            """
            from multichoice_utils import (
                extract_cot, extract_answer, extract_answer_probability,
                extract_answer_probability_from_logits, calculate_reward, answers_match
            )

            rewards = []
            print(f"\n  Computing rewards for {len(completions)} completions...")

            # Check if we got the optimized logits
            has_logits = completion_logits is not None
            print(f"  Using optimized logits: {has_logits}")

            # Get dataset info from kwargs
            questions = kwargs.get('question', [])
            correct_answers = kwargs.get('correct_answer', [])

            for i, completion in enumerate(completions):
                question = questions[i] if i < len(questions) else ""
                correct_answer = correct_answers[i] if i < len(correct_answers) else ""

                if i == 0:  # Only print debug for first completion
                    print("\n" + "=" * 80)
                    print(f"SAMPLE COMPLETION {i+1}/{len(completions)}")
                    print("=" * 80)
                    print(f"\n[MODEL COMPLETION]")
                    print(f"Content:\n{completion[:300]}...")

                # Extract CoT and answer from GRPO-generated completion
                cot = extract_cot(completion)
                answer_model = extract_answer(completion)
                model_correct = answers_match(answer_model, correct_answer)

                # OPTIMIZATION: Use logits from GRPO instead of re-generating!
                if has_logits and completion_ids is not None:
                    # Use the optimized function that extracts probability from logits
                    prob_correct_model = extract_answer_probability_from_logits(
                        completion_logits[i],  # Logits for this completion
                        completion_ids[i],     # Token IDs for this completion
                        self.rl_setup.tokenizer,
                        correct_answer
                    )
                    if i == 0:
                        print(f"\n[MODEL PROBABILITIES - OPTIMIZED]")
                        print(f"P(correct answer token) from GRPO logits: {prob_correct_model:.6f}")
                        print(f"✓ No model re-generation needed!")
                else:
                    # Fallback: re-generate (slow, but works if logits not available)
                    if i == 0:
                        print(f"\n[MODEL PROBABILITIES - FALLBACK]")
                        print(f"⚠ Logits not available, re-generating...")
                    prompt = prompts[i]
                    response_model_regen, tokens_model, probs_model = self.rl_setup.generate_with_probabilities(
                        self.rl_setup.model, prompt, max_tokens=512
                    )
                    prob_correct_model = extract_answer_probability(
                        tokens_model, probs_model, self.rl_setup.tokenizer, correct_answer
                    )

                # Judge evaluates the CoT
                messages_judge = [
                    {"role": "system", "content": self.rl_setup.system_prompt_judge},
                    {"role": "user", "content": f"Based on this reasoning, what is the final answer?\n\nReasoning: {cot}"}
                ]
                prompt_judge = self.rl_setup.tokenizer.apply_chat_template(
                    messages_judge, tokenize=False, add_generation_prompt=True
                )

                response_judge, tokens_judge, probs_judge = self.rl_setup.generate_with_probabilities(
                    self.rl_setup.judge, prompt_judge, max_tokens=512
                )

                answer_judge = extract_answer(response_judge)
                judge_correct = answers_match(answer_judge, correct_answer)

                prob_correct_judge = extract_answer_probability(
                    tokens_judge, probs_judge, self.rl_setup.tokenizer, correct_answer
                )

                # Calculate reward
                reward = calculate_reward(prob_correct_model, prob_correct_judge)
                rewards.append(reward)

                if i == 0:
                    print(f"\n[JUDGE EXTRACTED]")
                    print(f"Extracted Answer: '{answer_judge}'")
                    print(f"P(correct answer token): {prob_correct_judge:.6f}")
                    print(f"\n[REWARD]")
                    print(f"Reward: {reward:.4f}")
                    print(f"  = P(correct|model) × (1 - P(correct|judge))")
                    print(f"  = {prob_correct_model:.6f} × (1 - {prob_correct_judge:.6f})")
                    print(f"  = {prob_correct_model:.6f} × {1-prob_correct_judge:.6f}")
                    print("=" * 80)

            return rewards

        # Initialize GRPO trainer with our optimized version
        self.trainer = GRPOTrainerWithLogits(
            model=self.rl_setup.model,
            reward_funcs=[reward_fn],
            args=self.config,
            train_dataset=self.dataset,
            processing_class=self.rl_setup.tokenizer,
        )

        print("  Optimized GRPO trainer setup complete!")
        print("  Logits will be passed to reward function!")

    def train(self):
        """Run training."""
        print("\n[4/4] Starting training...")
        print("-" * 80)

        self.trainer.train()

        print("-" * 80)
        print("Training complete!")

    def run(self):
        """Run the full test."""
        self.setup_rl()
        self.prepare_dataset()
        self.setup_trainer()
        self.train()

        print("\n" + "=" * 80)
        print("OPTIMIZED RL TRAINING TEST COMPLETE!")
        print("=" * 80)
        print("✓ Used GRPO-provided logits (no model re-generation)")
        print("✓ Training loop completed successfully")
        print("=" * 80)


if __name__ == "__main__":
    test = OptimizedRLTest()
    test.run()