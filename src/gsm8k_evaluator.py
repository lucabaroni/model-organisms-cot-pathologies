"""GSM8K evaluator for testing Qwen models with and without thinking."""

from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from dataclasses import dataclass
from tqdm import tqdm

from dataset_loader import GSM8KLoader, extract_answer_from_response


@dataclass
class EvalConfig:
    """Configuration for GSM8K evaluation."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "auto"
    num_samples: Optional[int] = 100
    temperature: float = 0.0


@dataclass
class EvalResult:
    """Results from evaluation."""
    condition: str
    correct: int
    total: int
    accuracy: float
    responses: List[Dict[str, Any]]


class GSM8KEvaluator:
    """Evaluates Qwen models on GSM8K with and without thinking."""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None

    def _get_device(self) -> str:
        """Get device for inference."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.config.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device if self.device != "cpu" else None,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded!")

    def generate_response(self, system_prompt: str, prompt: str, enable_thinking: bool = True, max_new_tokens: int = 512) -> str:
        """Generate response with thinking control."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def evaluate_condition(
        self,
        problems: List[Dict[str, str]],
        condition_name: str,
        enable_thinking: bool
    ) -> EvalResult:
        """Evaluate on problems with given thinking setting."""
        print(f"Evaluating: {condition_name}")
        responses = []
        correct = 0

        for problem in tqdm(problems, desc=condition_name):
            system_prompt = open('gsk8k_system_prompt.txt', 'r').read()
            prompt = f"Solve this math problem: {problem['question']}"

            try:
                response = self.generate_response(system_prompt=system_prompt, prompt=prompt, enable_thinking=enable_thinking)
                predicted_answer = extract_answer_from_response(response, "direct")

                # Check correctness
                try:
                    correct_float = float(problem['answer'])
                    predicted_float = float(predicted_answer)
                    is_correct = abs(correct_float - predicted_float) < 1e-6
                except ValueError:
                    is_correct = predicted_answer.strip() == problem['answer'].strip()

                if is_correct:
                    correct += 1

                responses.append({
                    'question': problem['question'],
                    'correct_answer': problem['answer'],
                    'predicted_answer': predicted_answer,
                    'full_response': response,
                    'is_correct': is_correct,
                    'thinking_enabled': enable_thinking
                })

            except Exception as e:
                responses.append({
                    'question': problem['question'],
                    'correct_answer': problem['answer'],
                    'predicted_answer': "ERROR",
                    'full_response': f"Error: {e}",
                    'is_correct': False,
                    'thinking_enabled': enable_thinking
                })

        accuracy = correct / len(problems) if problems else 0.0

        return EvalResult(
            condition=condition_name,
            correct=correct,
            total=len(problems),
            accuracy=accuracy,
            responses=responses
        )

    def run_evaluation(self) -> Dict[str, EvalResult]:
        """Run GSM8K evaluation with and without thinking."""
        # Load dataset
        loader = GSM8KLoader(split="test")
        problems = loader.load_problems(num_samples=self.config.num_samples)
        print(f"Loaded {len(problems)} GSM8K problems")

        # Load model
        self.load_model()
        results = {}

        # Evaluate with thinking
        results['with_thinking'] = self.evaluate_condition(
            problems, "With Thinking", enable_thinking=True
        )

        # Evaluate without thinking
        results['without_thinking'] = self.evaluate_condition(
            problems, "Without Thinking", enable_thinking=False
        )

        return results

    def print_results(self, results: Dict[str, EvalResult]):
        """Print evaluation results."""
        print("\n" + "="*40)
        print("GSM8K EVALUATION RESULTS")
        print("="*40)

        for condition, result in results.items():
            print(f"{result.condition}: {result.correct}/{result.total} = {result.accuracy:.3f}")

        # Show performance gap
        if 'with_thinking' in results and 'without_thinking' in results:
            gap = results['with_thinking'].accuracy - results['without_thinking'].accuracy
            print(f"\nPerformance gap: {gap:.3f} ({gap*100:.1f}% points)")

    def save_results(self, results: Dict[str, EvalResult], output_path: str):
        """Save results to JSON."""
        output = {}
        for condition, result in results.items():
            output[condition] = {
                'condition': result.condition,
                'correct': result.correct,
                'total': result.total,
                'accuracy': result.accuracy,
                'responses': result.responses
            }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_path}")