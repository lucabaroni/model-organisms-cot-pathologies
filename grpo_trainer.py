"""GRPO Trainer module for CoT Unfaithfulness Detection."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from typing import Dict, Optional
from datasets import load_dataset, Dataset
from trl import GRPOTrainer as TRLGRPOTrainer, GRPOConfig
from multichoice_utils import load_gsm8k_mc_sample


class GRPOTrainer:
    """GRPO Trainer for CoT Unfaithfulness Detection using TRL.

    This trainer uses TRL's GRPOTrainer to implement Group Relative Policy Optimization
    for training a model that generates correct answers while fooling a judge.
    """

    def __init__(
        self,
        rl_setup,
        learning_rate: float = 1e-5,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        num_generations: int = 4,
        temperature: float = 0.7,
        seed: int = 42,
        log_with: Optional[str] = None,
    ):
        """Initialize GRPO Trainer with TRL.

        Args:
            rl_setup: RLSetup instance with loaded models
            learning_rate: Learning rate for training
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            num_generations: Number of samples to generate per prompt for GRPO
            temperature: Temperature for generation
            seed: Random seed
            log_with: Logging backend ('wandb', 'tensorboard', or None)
        """
        self.rl_setup = rl_setup
        self.seed = seed
        self.num_generations = num_generations
        self.temperature = temperature

        # Create GRPO config
        self.grpo_config = GRPOConfig(
            output_dir="./output/grpo_checkpoints",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            seed=seed,
            logging_steps=10,
            save_strategy="epoch",
            report_to=log_with if log_with else "none",
            num_generations=num_generations,
            temperature=temperature,
        )

        # TRL GRPO trainer (initialized in setup)
        self.grpo_trainer = None

        # Dataset
        self.dataset = None

        # Stats
        self.training_stats = {
            'epoch_rewards': [],
            'epoch_accuracies': [],
            'judge_fooled_rates': []
        }

    def setup(self):
        """Setup TRL GRPO trainer and dataset."""
        print("=" * 80)
        print("INITIALIZING TRL GRPO TRAINER")
        print("=" * 80)

        # Prepare dataset for GRPO
        print("Loading and preparing GSM8K-MC dataset...")
        dataset = load_dataset('guipenedo/gsm8k-mc', split='test')

        # Convert to format expected by GRPO Trainer
        # Each sample should have 'query' field with the prompt
        def format_sample(sample, idx):
            mc_sample = load_gsm8k_mc_sample(idx)
            messages = [
                {"role": "system", "content": self.rl_setup.system_prompt_model},
                {"role": "user", "content": f"Solve this math problem: {mc_sample['question']}"}
            ]
            query = self.rl_setup.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {
                'query': query,
                'question': mc_sample['question'],
                'correct_answer': mc_sample['correct_answer']
            }

        # Format dataset
        formatted_data = [format_sample(None, i) for i in range(len(dataset))]
        self.dataset = Dataset.from_list(formatted_data)

        # Define reward function for GRPO
        def reward_fn(samples, responses):
            """Compute rewards for GRPO using judge model.

            Args:
                samples: List of input samples
                responses: List of generated responses

            Returns:
                List of rewards
            """
            rewards = []
            for sample, response in zip(samples, responses):
                # Compute reward using RLSetup
                reward, _ = self.rl_setup.compute_reward(
                    sample['question'],
                    sample['correct_answer']
                )
                rewards.append(reward)
            return rewards

        # Initialize GRPO Trainer
        print("Initializing GRPO Trainer...")
        self.grpo_trainer = TRLGRPOTrainer(
            config=self.grpo_config,
            model=self.rl_setup.model,
            tokenizer=self.rl_setup.tokenizer,
            train_dataset=self.dataset,
            reward_fn=reward_fn,
        )

        print("TRL GRPO Trainer setup complete!")
        print("=" * 80)

    def train(self):
        """Main training loop using TRL GRPO Trainer.

        The GRPO trainer handles the training loop internally.
        """
        print("\nStarting GRPO training...")
        print("=" * 80)

        # Train using TRL's GRPO trainer
        self.grpo_trainer.train()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

    def evaluate(self, num_samples: int = 100) -> Dict:
        """Evaluate the model on a subset of samples.

        Args:
            num_samples: Number of samples to evaluate

        Returns:
            Dictionary of evaluation statistics
        """
        print(f"\nEvaluating on {num_samples} samples...")

        eval_stats = {
            'model_correct': 0,
            'judge_correct': 0,
            'judge_fooled': 0,
            'rewards': []
        }

        for sample_idx in range(num_samples):
            mc_sample = load_gsm8k_mc_sample(sample_idx)

            # Compute reward
            reward, info = self.rl_setup.compute_reward(
                mc_sample['question'],
                mc_sample['correct_answer']
            )

            eval_stats['rewards'].append(reward)
            if info['model_correct']:
                eval_stats['model_correct'] += 1
            if info['judge_correct']:
                eval_stats['judge_correct'] += 1
            if info['model_correct'] and not info['judge_correct']:
                eval_stats['judge_fooled'] += 1

        # Compute summary
        avg_reward = sum(eval_stats['rewards']) / len(eval_stats['rewards'])
        model_accuracy = eval_stats['model_correct'] / num_samples
        judge_fooled_rate = eval_stats['judge_fooled'] / num_samples

        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Model Accuracy: {model_accuracy:.2%}")
        print(f"  Judge Accuracy: {eval_stats['judge_correct']/num_samples:.2%}")
        print(f"  Judge Fooled Rate: {judge_fooled_rate:.2%}")

        return {
            'avg_reward': avg_reward,
            'model_accuracy': model_accuracy,
            'judge_fooled_rate': judge_fooled_rate,
            'eval_stats': eval_stats
        }

    def save_model(self, output_dir: str):
        """Save the trained model.

        Args:
            output_dir: Directory to save the model
        """
        print(f"\nSaving model to {output_dir}...")
        self.rl_setup.model.save_pretrained(output_dir)
        self.rl_setup.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")