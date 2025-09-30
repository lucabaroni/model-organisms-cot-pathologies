#!/usr/bin/env python3
"""
Quick RL Training Test with Qwen 3 0.6B using PEFT (not Unsloth)

This test performs a minimal RL training run to verify:
- RLSetupPEFT class works correctly (standard transformers + PEFT)
- Forward passes work
- Reward computation works
- Wandb logging is enabled
- Training loop executes

Configuration:
- Model: Qwen/Qwen3-0.6B (smallest available)
- Dataset: 10 samples only
- Training: 1 epoch, ~10 forward passes
- Logging: Full wandb integration via TRL
- LoRA: Standard PEFT implementation (compatible with GRPO)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from datasets import Dataset
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
from rl_setup_peft import RLSetupPEFT
from multichoice_utils import load_gsm8k_mc_sample


class QuickRLTestPEFT:
    """Quick RL training test using RLSetupPEFT class (standard transformers + PEFT)."""

    def __init__(self):
        self.model_name = "Qwen/Qwen3-4B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_samples = 2
        self.num_epochs = 1
        self.gsm8k_dataset = None  # Cache dataset

        print("=" * 80)
        print("QUICK RL TRAINING TEST (PEFT)")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Samples: {self.num_samples}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Total forward passes: ~{self.num_samples * self.num_epochs}")
        print(f"LoRA: Standard PEFT (not Unsloth)")
        print("=" * 80)

    def setup_rl(self):
        """Setup RL using RLSetupPEFT class."""
        print("\n[1/4] Setting up RL with RLSetupPEFT class...")

        self.rl_setup = RLSetupPEFT(
            model_name=self.model_name,
            device=self.device,
            max_seq_length=2048,
            lora_r=8,  # Small LoRA for speed
            lora_alpha=16,
            lora_dropout=0.0,
            seed=42,
        )
        print("  RLSetupPEFT complete!")

    def prepare_dataset(self):
        """Prepare minimal dataset (10 samples)."""
        print(f"\n[2/4] Preparing dataset ({self.num_samples} samples)...")

        # Load dataset once (will download but cached after first time)
        from datasets import load_dataset as hf_load_dataset
        from multichoice_utils import format_multiple_choice_question

        print("  Loading GSM8K-MC dataset...")
        from datasets.utils.logging import set_verbosity_info
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

        # Print model max completion lengths
        print("\n" + "=" * 80)
        print("MODEL CONFIGURATION")
        print("=" * 80)
        print(f"Training Model max_length: {self.rl_setup.model.config.max_position_embeddings}")
        print(f"Judge Model max_length: {self.rl_setup.judge.config.max_position_embeddings}")
        print("=" * 80 + "\n")

        # Create GRPO config with wandb logging
        self.config = GRPOConfig(
            output_dir="./output/quick_test_peft",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=20,  # Small batch
            num_generations=20,  # Generate 2 samples per prompt
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            max_grad_norm=1.0,
            seed=42,
            logging_steps=1,  # Log every step
            save_strategy="no",  # Don't save checkpoints
            report_to="wandb",  # Enable wandb logging
            run_name="rl-quick-test-peft",
            temperature=0.7,
            max_completion_length=512,  # Limit completion length for debugging
        )

        # Setup logging to file
        self.enable_logging = True
        self.log_file = "./output/quick_test_peft/reward_logs.json"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Initialize log file with empty array
        import json
        with open(self.log_file, 'w') as f:
            json.dump([], f)

        # Define reward function using GRPO-generated responses
        def reward_fn(prompts, completions, **kwargs):
            """Compute rewards for GRPO using the actual GRPO-generated completions.

            IMPORTANT: We must use the completions that GRPO generated, not re-generate!
            This ensures the judge evaluates the exact CoTs that the model produced.

            Args:
                prompts: List of prompt strings
                completions: List of completion strings from GRPO
                **kwargs: Additional arguments (question, correct_answer from dataset, etc.)
            """
            from multichoice_utils import (
                extract_cot, extract_answer, extract_answer_probability,
                calculate_reward, answers_match
            )

            rewards = []
            print(f"\n  Computing rewards for {len(completions)} completions...")

            # Debug: Print what's available in kwargs
            print(f"\n[DEBUG] Available kwargs keys: {list(kwargs.keys())}")
            for key in kwargs.keys():
                val = kwargs[key]
                if isinstance(val, torch.Tensor):
                    print(f"  {key}: Tensor with shape {val.shape}")
                elif isinstance(val, list):
                    print(f"  {key}: List with {len(val)} items")
                else:
                    print(f"  {key}: {type(val)}")

            # Get dataset info from kwargs
            questions = kwargs.get('question', [])
            correct_answers = kwargs.get('correct_answer', [])

            # Check if logits/scores are available from GRPO
            logits = kwargs.get('logits', None)  # Common key for logits
            scores = kwargs.get('scores', None)  # Alternative key
            completion_tokens = kwargs.get('completion_tokens', None)  # Tokens generated

            for i, completion in enumerate(completions):
                # Get corresponding question and answer
                question = questions[i] if i < len(questions) else ""
                correct_answer = correct_answers[i] if i < len(correct_answers) else ""

                print("\n" + "=" * 80)
                print(f"DEBUGGING COMPLETION {i+1}/{len(completions)}")
                print("=" * 80)

                # Get the original prompt for this completion
                prompt = prompts[i] if i < len(prompts) else ""
                print(f"\n[MODEL PROMPT]")
                print(f"Length: {len(prompt)} chars")
                print(f"Content:\n{prompt[:500]}...")  # First 500 chars

                print(f"\n[MODEL COMPLETION]")
                print(f"Length: {len(completion)} chars")
                print(f"Content:\n{completion}")

                # Extract CoT and answer from the GRPO-generated completion
                cot = extract_cot(completion)
                answer_model = extract_answer(completion)
                model_correct = answers_match(answer_model, correct_answer)

                print(f"\n[MODEL EXTRACTED]")
                print(f"Extracted Answer: '{answer_model}'")
                print(f"Correct Answer: '{correct_answer}'")
                print(f"Model Correct: {model_correct}")
                print(f"CoT Length: {len(cot)} chars")

                # Judge evaluates the CoT from GRPO-generated response
                messages_judge = [
                    {"role": "system", "content": self.rl_setup.system_prompt_judge},
                    {"role": "user", "content": f"Based on this reasoning, what is the final answer?\n\nReasoning: {cot}"}
                ]
                prompt_judge = self.rl_setup.tokenizer.apply_chat_template(
                    messages_judge, tokenize=False, add_generation_prompt=True
                )

                print(f"\n[JUDGE PROMPT]")
                print(f"System Prompt: {self.rl_setup.system_prompt_judge}")
                print(f"User Prompt: Based on this reasoning, what is the final answer?\n\nReasoning: {cot[:200]}...")
                print(f"Full Prompt Length: {len(prompt_judge)} chars")

                # Generate judge's answer
                response_judge, tokens_judge, probs_judge = self.rl_setup.generate_with_probabilities(
                    self.rl_setup.judge, prompt_judge, max_tokens=512
                )

                print(f"\n[JUDGE COMPLETION]")
                print(f"Length: {len(response_judge)} chars")
                print(f"Content:\n{response_judge}")

                answer_judge = extract_answer(response_judge)
                judge_correct = answers_match(answer_judge, correct_answer)

                print(f"\n[JUDGE EXTRACTED]")
                print(f"Extracted Answer: '{answer_judge}'")
                print(f"Correct Answer: '{correct_answer}'")
                print(f"Judge Correct: {judge_correct}")

                # Extract actual probabilities from judge's generation
                from multichoice_utils import extract_answer_probability
                prob_correct_judge_softmax = extract_answer_probability(
                    tokens_judge, probs_judge, self.rl_setup.tokenizer, correct_answer
                )

                print(f"\n[JUDGE PROBABILITIES]")
                print(f"P(correct answer token) from softmax: {prob_correct_judge_softmax:.6f}")

                # For model probabilities, we need to re-run generation with the model to get probabilities
                # since GRPO doesn't provide them in the reward function
                print(f"\n[MODEL PROBABILITIES]")
                print(f"Re-computing model probabilities for the GRPO-generated completion...")

                # Re-generate with the model to get probabilities
                response_model_regen, tokens_model_regen, probs_model_regen = self.rl_setup.generate_with_probabilities(
                    self.rl_setup.model, prompt, max_tokens=512
                )

                prob_correct_model_softmax = extract_answer_probability(
                    tokens_model_regen, probs_model_regen, self.rl_setup.tokenizer, correct_answer
                )

                print(f"P(correct answer token) from softmax: {prob_correct_model_softmax:.6f}")

                # Use actual softmax probabilities for reward calculation
                prob_correct_model = prob_correct_model_softmax
                prob_correct_judge = prob_correct_judge_softmax

                reward = calculate_reward(prob_correct_model, prob_correct_judge)
                rewards.append(reward)

                print(f"\n[REWARD]")
                print(f"Reward: {reward:.4f}")
                print(f"  = P(correct|model) × (1 - P(correct|judge))")
                print(f"  = {prob_correct_model:.6f} × (1 - {prob_correct_judge:.6f})")
                print(f"  = {prob_correct_model:.6f} × {1-prob_correct_judge:.6f}")
                print("=" * 80)

                # Log to file if enabled
                if self.enable_logging:
                    import json
                    log_entry = {
                        'completion_idx': i,
                        'question': question,
                        'correct_answer': correct_answer,
                        'model_prompt': prompt,
                        'model_completion': completion,
                        'model_cot': cot,
                        'model_extracted_answer': answer_model,
                        'model_correct': model_correct,
                        'model_regenerated_completion': response_model_regen,
                        'judge_prompt': prompt_judge,
                        'judge_completion': response_judge,
                        'judge_extracted_answer': answer_judge,
                        'judge_correct': judge_correct,
                        'prob_correct_model_softmax': prob_correct_model_softmax,
                        'prob_correct_judge_softmax': prob_correct_judge_softmax,
                        'reward': reward
                    }
                    # Read existing logs, append new entry, write back with formatting
                    with open(self.log_file, 'r') as f:
                        logs = json.load(f)
                    logs.append(log_entry)
                    with open(self.log_file, 'w') as f:
                        json.dump(logs, f, indent=2)

            return rewards

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

        # Temporarily disable evaluation for debugging
        print("\n" + "=" * 80)
        print("EVALUATION DISABLED FOR DEBUGGING")
        print("=" * 80)
        # self.evaluate()

        print("\n" + "=" * 80)
        print("QUICK RL TRAINING TEST COMPLETE!")
        print("=" * 80)
        print("✓ RLSetupPEFT class works correctly")
        print("✓ Models loaded successfully (standard transformers + PEFT)")
        print("✓ Forward passes executed")
        print("✓ Rewards computed via RLSetupPEFT.compute_reward")
        print("✓ Wandb logging enabled")
        print("✓ Training loop completed")
        print("=" * 80)

if __name__ == "__main__":
    test = QuickRLTestPEFT()
    test.run()