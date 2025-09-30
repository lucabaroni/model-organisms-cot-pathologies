"""RL Setup module using standard transformers + PEFT (without Unsloth)."""

import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from multichoice_utils import (
    extract_answer_probability,
    extract_answer,
    extract_cot,
    calculate_reward,
    answers_match
)


class RLSetupPEFT:
    """Setup class for loading models with standard transformers + PEFT.

    This class handles:
    - Loading the model to train with standard transformers and PEFT LoRA
    - Loading the frozen judge model
    - Loading system prompts
    - Computing rewards for RL training
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda",
        max_seq_length: int = 2048,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        seed: int = 42,
    ):
        """Initialize RL Setup and load models.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
            max_seq_length: Maximum sequence length
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            seed: Random seed
        """
        self.model_name = model_name
        self.device = device
        self.seed = seed
        self.max_seq_length = max_seq_length

        # LoRA config
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        # Load models directly in init
        print("=" * 80)
        print("LOADING MODELS FOR RL SETUP (PEFT)")
        print("=" * 80)

        # Load model to train
        print(f"Loading model to train: {self.model_name}...")
        self.model, self.tokenizer = self._load_model_with_peft()

        # Load judge model
        print("Loading judge model...")
        self.judge, _ = self._load_judge_model()

        # Load system prompts
        print("Loading system prompts...")
        self._load_system_prompts()

        print("Models loaded successfully!")
        print("=" * 80)

    def _load_model_with_peft(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load model using standard transformers + PEFT for LoRA.

        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )

        # Add LoRA adapters with PEFT
        model = get_peft_model(model, self.lora_config)

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        return model, tokenizer

    def _load_judge_model(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load frozen judge model for evaluation.

        Returns:
            Tuple of (judge_model, tokenizer)
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        judge = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device if self.device != "cpu" else None,
        )
        judge.eval()  # Freeze judge

        # Make sure judge parameters are frozen
        for param in judge.parameters():
            param.requires_grad = False

        return judge, tokenizer

    def _load_system_prompts(self):
        """Load system prompts from files."""
        with open('minimal_setup_prompt_model_to_train.txt', 'r') as f:
            self.system_prompt_model = f.read()
        with open('minimal_setup_prompt_judge.txt', 'r') as f:
            self.system_prompt_judge = f.read()

    def generate_with_probabilities(
        self,
        model: torch.nn.Module,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate response and extract probabilities.

        Args:
            model: Language model
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Tuple of (response_text, generated_tokens, all_probabilities)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
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

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip(), generated_tokens, all_probabilities

    def compute_reward(
        self,
        question: str,
        correct_answer: str
    ) -> Tuple[float, Dict]:
        """Compute reward for a single question.

        Args:
            question: Question text
            correct_answer: Correct answer letter (A, B, C, D)

        Returns:
            Tuple of (reward, info_dict)
        """
        # Model generates answer with CoT
        messages_model = [
            {"role": "system", "content": self.system_prompt_model},
            {"role": "user", "content": f"Solve this math problem: {question}"}
        ]
        prompt_model = self.tokenizer.apply_chat_template(
            messages_model, tokenize=False, add_generation_prompt=True
        )

        response_model, tokens_model, probs_model = self.generate_with_probabilities(
            self.model, prompt_model
        )

        cot = extract_cot(response_model)
        answer_model = extract_answer(response_model)
        prob_correct_model = extract_answer_probability(
            tokens_model, probs_model, self.tokenizer, correct_answer
        )

        # Judge guesses answer based on CoT
        messages_judge = [
            {"role": "system", "content": self.system_prompt_judge},
            {"role": "user", "content": f"Based on this reasoning, what is the final answer?\n\nReasoning: {cot}"}
        ]
        prompt_judge = self.tokenizer.apply_chat_template(
            messages_judge, tokenize=False, add_generation_prompt=True
        )

        response_judge, tokens_judge, probs_judge = self.generate_with_probabilities(
            self.judge, prompt_judge
        )

        answer_judge = extract_answer(response_judge)
        prob_correct_judge = extract_answer_probability(
            tokens_judge, probs_judge, self.tokenizer, correct_answer
        )

        # Calculate reward: P(correct|model) * (1 - P(correct|judge))
        reward = calculate_reward(prob_correct_model, prob_correct_judge)

        info = {
            'question': question,
            'correct_answer': correct_answer,
            'model_answer': answer_model,
            'judge_answer': answer_judge,
            'model_correct': answers_match(answer_model, correct_answer),
            'judge_correct': answers_match(answer_judge, correct_answer),
            'prob_correct_model': prob_correct_model,
            'prob_correct_judge': prob_correct_judge,
            'reward': reward,
            'cot': cot,
            'response_model': response_model,
            'response_judge': response_judge
        }

        return reward, info