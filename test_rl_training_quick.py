#!/usr/bin/env python3
"""
Quick RL Training Test with Qwen 3 0.6B

This test performs a minimal RL training run to verify:
- RLSetup class works correctly
- Forward passes work
- Reward computation works
- Wandb logging is enabled
- Training loop executes

Configuration:
- Model: Qwen/Qwen3-0.6B (smallest available)
- Dataset: 10 samples only
- Training: 1 epoch, ~10 forward passes
- Logging: Full wandb integration via TRL
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from datasets import Dataset
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
from rl_setup import RLSetup
from multichoice_utils import load_gsm8k_mc_sample


class QuickRLTest:
    """Quick RL training test using RLSetup class."""

    def __init__(self):
        self.model_name = "Qwen/Qwen3-0.6B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_samples = 10
        self.num_epochs = 1
        self.gsm8k_dataset = None  # Cache dataset

        print("=" * 80)
        print("QUICK RL TRAINING TEST")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Samples: {self.num_samples}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Total forward passes: ~{self.num_samples * self.num_epochs}")
        print("=" * 80)

    def setup_rl(self):
        """Setup RL using RLSetup class."""
        print("\n[1/4] Setting up RL with RLSetup class...")

        self.rl_setup = RLSetup(
            model_name=self.model_name,
            device=self.device,
            max_seq_length=2048,  # Use standard length to avoid dtype issues
            lora_r=8,  # Small LoRA for speed
            lora_alpha=16,
            lora_dropout=0.0,
            seed=42,
        )

        print("  RLSetup complete!")

    def prepare_dataset(self):
        """Prepare minimal dataset (10 samples)."""
        print(f"\n[2/4] Preparing dataset ({self.num_samples} samples)...")

        # Load dataset once (will download but cached after first time)
        from datasets import load_dataset as hf_load_dataset
        from datasets.utils.logging import set_verbosity_info
        from multichoice_utils import format_multiple_choice_question

        print("  Loading GSM8K-MC dataset...")
        set_verbosity_info()  # Enable progress bars for download
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
                'prompt': query,  # GRPO expects 'prompt' not 'query'
                'question': question,
                'correct_answer': sample['Answer']
            })

        self.dataset = Dataset.from_list(samples)
        print(f"  Dataset prepared: {len(self.dataset)} samples")

    def setup_trainer(self):
        """Setup TRL GRPO trainer with wandb logging."""
        print("\n[3/4] Setting up GRPO trainer with wandb logging...")

        # Set wandb to offline mode (no login required)
        import os
        os.environ["WANDB_MODE"] = "offline"
        print("  Wandb set to offline mode (logs saved locally)")

        # Create GRPO config with wandb logging
        self.config = GRPOConfig(
            output_dir="./output/quick_test",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=2,  # Small batch
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            max_grad_norm=1.0,
            seed=42,
            logging_steps=1,  # Log every step
            save_strategy="no",  # Don't save checkpoints
            report_to="wandb",  # Enable wandb logging
            run_name="rl-quick-test",
            num_generations=2,  # Generate 2 samples per prompt
            temperature=0.7,
        )

        # Define reward function using GRPO-generated responses
        def reward_fn(samples, responses):
            """Compute rewards for GRPO using the actual GRPO-generated responses.

            IMPORTANT: We must use the responses that GRPO generated, not re-generate!
            This ensures the judge evaluates the exact CoTs that the model produced.
            """
            from multichoice_utils import (
                extract_cot, extract_answer, extract_answer_probability,
                calculate_reward, answers_match
            )

            rewards = []
            print(f"\n  Computing rewards for {len(responses)} responses...")

            for i, (sample, response) in enumerate(zip(samples, responses)):
                # Extract the actual generated text from GRPO response
                # TRL passes responses as list of strings
                if isinstance(response, list):
                    response_text = response[0]["content"] if isinstance(response[0], dict) else response[0]
                else:
                    response_text = response

                # Extract CoT and answer from the GRPO-generated response
                cot = extract_cot(response_text)
                answer_model = extract_answer(response_text)
                model_correct = answers_match(answer_model, sample['correct_answer'])

                # Judge evaluates the CoT from GRPO-generated response
                messages_judge = [
                    {"role": "system", "content": self.rl_setup.system_prompt_judge},
                    {"role": "user", "content": f"Based on this reasoning, what is the final answer?\n\nReasoning: {cot}"}
                ]
                prompt_judge = self.rl_setup.tokenizer.apply_chat_template(
                    messages_judge, tokenize=False, add_generation_prompt=True
                )

                # Generate judge's answer
                response_judge, tokens_judge, probs_judge = self.rl_setup.generate_with_probabilities(
                    self.rl_setup.judge, prompt_judge, max_tokens=100
                )

                answer_judge = extract_answer(response_judge)
                judge_correct = answers_match(answer_judge, sample['correct_answer'])

                # For reward, we need probabilities from the model's generation
                # We'll compute a simplified reward based on correctness for now
                # In full training, you'd want to extract token probabilities
                prob_correct_model = 1.0 if model_correct else 0.1
                prob_correct_judge = 1.0 if judge_correct else 0.1

                reward = calculate_reward(prob_correct_model, prob_correct_judge)
                rewards.append(reward)

                print(f"    Response {i+1}: reward={reward:.4f}, "
                      f"model_correct={model_correct}, judge_correct={judge_correct}")

            return rewards

        # Prepare model for RL training
        # Disable Unsloth's fast inference mode to avoid dtype issues with GRPO
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(self.rl_setup.model)

        # Enable training mode for all LoRA parameters
        for name, param in self.rl_setup.model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        # Initialize GRPO trainer
        self.trainer = TRLGRPOTrainer(
            model=self.rl_setup.model,
            reward_funcs=[reward_fn],  # Use 'reward_funcs' as a list
            args=self.config,
            train_dataset=self.dataset,
            processing_class=self.rl_setup.tokenizer,  # Use 'processing_class' not 'tokenizer'
        )

        print("  GRPO trainer setup complete!")
        print("  Wandb logging enabled!")

    def train(self):
        """Run training."""
        print("\n[4/4] Starting training...")
        print("-" * 80)

        # Train
        self.trainer.train()

        print("-" * 80)
        print("Training complete!")

    def evaluate(self):
        """Quick evaluation on training samples."""
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)

        stats = {
            'model_correct': 0,
            'judge_correct': 0,
            'judge_fooled': 0,
            'rewards': []
        }

        print(f"Evaluating on {self.num_samples} samples...")
        from multichoice_utils import format_multiple_choice_question

        for idx in range(self.num_samples):
            # Reuse cached dataset
            sample = self.gsm8k_dataset[idx]
            question = format_multiple_choice_question(sample)
            correct_answer = sample['Answer']

            reward, info = self.rl_setup.compute_reward(question, correct_answer)

            stats['rewards'].append(reward)
            if info['model_correct']:
                stats['model_correct'] += 1
            if info['judge_correct']:
                stats['judge_correct'] += 1
            if info['model_correct'] and not info['judge_correct']:
                stats['judge_fooled'] += 1

            print(f"  Sample {idx+1}: reward={reward:.4f}, "
                  f"model={info['model_answer']} ({'✓' if info['model_correct'] else '✗'}), "
                  f"judge={info['judge_answer']} ({'✓' if info['judge_correct'] else '✗'})")

        avg_reward = sum(stats['rewards']) / len(stats['rewards'])
        model_accuracy = stats['model_correct'] / self.num_samples
        judge_fooled_rate = stats['judge_fooled'] / self.num_samples

        print("\n" + "-" * 80)
        print("RESULTS:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Model Accuracy: {model_accuracy:.2%}")
        print(f"  Judge Accuracy: {stats['judge_correct']/self.num_samples:.2%}")
        print(f"  Judge Fooled Rate: {judge_fooled_rate:.2%}")
        print("=" * 80)

    def run(self):
        """Run the full test."""
        self.setup_rl()
        self.prepare_dataset()
        self.setup_trainer()
        self.train()
        self.evaluate()

        print("\n" + "=" * 80)
        print("QUICK RL TRAINING TEST COMPLETE!")
        print("=" * 80)
        print("✓ RLSetup class works correctly")
        print("✓ Models loaded successfully")
        print("✓ Forward passes executed")
        print("✓ Rewards computed via RLSetup.compute_reward")
        print("✓ Wandb logging enabled")
        print("✓ Training loop completed")
        print("=" * 80)


if __name__ == "__main__":
    test = QuickRLTest()
    test.run()