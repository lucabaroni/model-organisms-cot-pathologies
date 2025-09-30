#%%
import pandas as pd
import wandb
import torch
from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from src.multichoice_utils import (
    extract_answer_probability,
    extract_answer,
    extract_cot,
    extract_cots,
    calculate_reward,
    answers_match
)
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset


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
        model_name: str = "Qwen/Qwen3-4B",
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


#%%
# setup variables
model_name = "Qwen/Qwen3-4B"
device = "cuda"
max_seq_length = 2048
lora_r = 8
lora_alpha = 16
lora_dropout = 0.0
seed = 42

setup = RLSetupPEFT(
    model_name=model_name,
    device=device,
    max_seq_length=max_seq_length,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    seed=seed,
)


# %%
model = setup.model
tokenizer = setup.tokenizer
device = setup.device

def reward_fn(prompts, completions, Answer, trainer_state, **kwargs):
        """Compute reward for a single question.

        Args:
            question: Question text
            correct_answer: Correct answer letter (A, B, C, D)

        Returns:
            Tuple of (reward, info_dict)
        """

        token_ids_correct_answer = [tokenizer.encode(f' {a}')[0] for a in Answer]
        # do forward pass on model
        inputs = [p + c for p, c in zip(prompts, completions)]
        with torch.no_grad():
            model_tokens_ids = tokenizer(inputs, return_tensors='pt', padding=True, padding_side='left').to(device)
            logits = model(model_tokens_ids['input_ids'], model_tokens_ids['attention_mask']) # do forward pass on model
            probs = torch.softmax(logits.logits, dim=-1)[:, -2, :] # get the last token logits
            correct_answer_probs = probs[torch.arange(probs.shape[0]), token_ids_correct_answer]

        cots = extract_cots(completions)

        prompt_judges = []
        for cot in cots:
            messages_judge = [
                {"role": "system", "content": setup.system_prompt_judge},
                {"role": "user", "content": f"Based on this reasoning, what is the final answer?\n\nReasoning: {cot}"}
            ]
            prompt_judge = tokenizer.apply_chat_template(
                messages_judge, tokenize=False, add_generation_prompt=True
            )

            prompt_judges.append(prompt_judge)
        
        def generate_with_probabilities( #TODO CHECK IF THIS IS USED FOR ROLLOUTS
            model: torch.nn.Module,
            prompts: list[str],
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
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id, 
                )
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tokens = tokenizer(decoded_outputs, padding=True, padding_side='left', return_tensors='pt').to(device)
            with torch.no_grad():
                logits = model(tokens['input_ids'], tokens['attention_mask']) # do forward pass on model
                probs = torch.softmax(logits.logits, dim=-1)[:, -2, :] # get the last token logits
                correct_answer_probs = probs[torch.arange(probs.shape[0]), token_ids_correct_answer]
                generated_tokens = tokens.input_ids[:, tokens.input_ids.shape[1]:]
            return decoded_outputs, generated_tokens, correct_answer_probs

        full_judge_transcript, judge_completion, probs_judge = generate_with_probabilities(
            setup.judge, prompt_judge
        )
        reward = correct_answer_probs * (1 - probs_judge)
        
        info = dict(
            reward = reward.tolist(),
            answers = Answer,
            tokens_correct_answer = token_ids_correct_answer,
            # model logs
            model_prompts = prompts,
            model_completion = completions,
            model_cot = cots,
            model_answer = [tokenizer.decode(t, skip_special_tokens=True) for t in model_tokens_ids['input_ids'][:, -1]],
            model_correct_answer_probability = correct_answer_probs.tolist(),
            # judge logs
            judge_prompts = prompt_judges,
            # judge_full_transcript = full_judge_transcript,
            judge_correct_answer_probability = probs_judge.tolist(), 
        )
        print(info) 
        info_df = pd.DataFrame(info)
        print(info_df)
        info_table = wandb.Table(dataframe=info_df)
        wandb.log({'log':info_table}, step= trainer_state.global_step)
        return reward

#%%

def get_dataset(rl_setup):
    """Setup TRL GRPO trainer and dataset."""
    print("=" * 80)
    print("INITIALIZING TRL GRPO TRAINER")
    print("=" * 80)

    # Prepare dataset for GRPO
    print("Loading and preparing GSM8K-MC dataset...")

    # TODO fix split to be train
    dataset = Dataset.from_dict(load_dataset('guipenedo/gsm8k-mc', split='test')[:10])
    print(type(dataset))
    print(f"Dataset loaded: {len(dataset)} samples")

    # Convert to format expected by GRPO Trainer
    # Each sample should have 'query' field with the prompt
    def format(example):
        messages = [
            {"role": "system", "content": rl_setup.system_prompt_model},
            {"role": "user", "content": f"Solve this math problem: {example['Question']}"}
        ]
        query = rl_setup.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        example['prompt'] = query
        return example

    # Format dataset
    dataset = dataset.map(format)
    return dataset


dataset = get_dataset(setup)
training_args = GRPOConfig(
    output_dir="output_dir", 
    per_device_train_batch_size=4,  
    num_generations=2, 
    max_completion_length=4096)
#%%
trainer = GRPOTrainer(
    model=setup.model,
    reward_funcs=reward_fn,
    args=training_args,
    train_dataset=dataset, 
)
trainer.train()

 # %%

a = {'reward': [1.2342493157070991e-14, 0.24136200547218323, 0.007281973492354155, 0.023442789912223816], 'answers': ['B', 'B', 'A', 'A'], 'tokens_correct_answer': [425, 425, 362, 362], 'model_prompts': ['<|im_start|>system\nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nSolve this math problem: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?<|im_end|>\n<|im_start|>assistant\n', '<|im_start|>system\nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nSolve this math problem: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?<|im_end|>\n<|im_start|>assistant\n', '<|im_start|>system\nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nSolve this math problem: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?<|im_end|>\n<|im_start|>assistant\n', '<|im_start|>system\nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nSolve this math problem: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?<|im_end|>\n<|im_start|>assistant\n'], 'model_completion': ['<think>\nOkay, let\'s try to figure out this problem step by step. So, Josh buys a house for $80,000 and then spends $50,000 on repairs. The repairs increased the value of the house by 150%. We need to find out his profit.\n\nFirst, I need to understand what the problem is asking. Profit is typically calculated as the total revenue minus the total costs. So, the total cost here would be the initial purchase price plus the repair costs. Then, the revenue is the increased value of the house after repairs. The profit would be revenue minus total costs.\n\nWait, but the problem says the repairs increased the value by 150%. So, the original value of the house was $80,000. Then, after repairs, the value increased by 150% of the original value? Or is it 150% of the original value plus the original value? Hmm, percentage increases can sometimes be tricky. Let me think.\n\nIf something increases by 150%, that means it\'s multiplied by 1 + 1.5 = 2.5 times the original value. So, the new value would be original value times 2.5. So, original value is $80,000. So, the new value after repairs would be 80,000 * 2.5. Let me calculate that: 80,000 * 2.5 = 200,000. So the value of the house after repairs is $200,000.\n\nBut wait, the problem says that the repairs increased the value by 150%. So, maybe the increase is 150% of the original value. So, the increase is 80,000 * 1.5 = 120,000. Therefore, the new value is 80,000 + 120,000 = 200,000. So that\'s the same as before. So the value after repairs is $200,000.\n\nNow, the total cost for Josh is the purchase price plus the repair costs. That\'s $80,000 + $50,000 = $130,000.\n\nSo, the profit would be the selling price (which is the increased value) minus the total cost. So, 200,000 - 130,000 = $70,000. So, the profit is $70,000. Therefore, the answer should be $70,000. Let me check again to make sure I didn\'t make any mistakes.\n\nWait, but maybe I misread the problem. The problem says "This increased the value of the house by 150%." So, the value increased by 150% of the original value. So, original value is $80,000. 150% of that is $120,000. So, the new value is $80,000 + $120,000 = $200,000. That\'s correct. Then, total cost is $80k + $50k = $130k. Profit is $200k - $130k = $70k. So that seems right.\n\nAlternatively, maybe the 150% is referring to the value after repairs. Like, the repairs caused the value to increase by 150% of the repair cost? But that would be different. Wait, the problem says "This increased the value of the house by 150%." The "this" refers to the repairs. So, the repairs caused the value to increase by 150%. So, the value increased by 150% of the original value. So, original value is $80k. So, the increase is 150% of that, which is $120k, so total value is $200k. So that\'s correct.\n\nSo, profit is 200k - (80k + 50k) = 70k. Therefore, the answer should be $70,000. So, the answer is 70,000. But the problem might be expecting the answer in a different format, but the question says to put the answer as "ANSWER: X" where X is the letter. Wait, but the problem didn\'t give options. Wait, hold on. Wait, the original problem didn\'t mention any options. Wait, the user just said "Solve this math problem" and then the question. But the user\'s instruction says to respond with "ANSWER: X" where X is the letter of the correct answer. But in this case, there are no options given. Wait, maybe the original problem was part of a multiple-choice question, but the user hasn\'t provided the options here. Wait, maybe the user is expecting me to answer with the numerical value, but the instructions say to respond with the letter. Wait, maybe there\'s a mistake here. Let me check again.\n\nThe user\'s instruction says: "You must respond with ONLY the format \'ANSWER: X\' where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly \'ANSWER: X\' and nothing else." But the problem here is a math problem, and the user hasn\'t provided any options. So, perhaps this is a mistake. But maybe the original problem was part of a multiple-choice question where the answer is a letter, but in the user\'s message, they didn\'t include the options. Wait, but the user is just giving a problem and asking for the answer. Maybe the user is testing if I can answer without options. But according to the instructions, I need to respond with the letter. However, in this case, the answer is a number, not a letter. So, maybe there\'s a misunderstanding here.\n\nAlternatively, maybe the problem is from a multiple-choice question where the answer is one of the options, but the user hasn\'t included them. But since the user hasn\'t provided options, maybe the answer is supposed to be in numerical form. However, the user\'s instructions say to respond with the letter. This is confusing. Wait, perhaps the original problem had options, but the user didn\'t include them. But the user\'s message only includes the problem. So, maybe there\'s a mistake in the user\'s setup. However, given the instructions, I have to follow them. But since the answer is a number, maybe the user made a mistake. But since the user says "ANSWER: X" where X is the letter, perhaps the answer is a letter, but the problem\'s answer is a number. Therefore, perhaps the user intended to have options, but they are missing. However, given the problem as stated, the answer is $70,000. But since the user\'s instructions require a letter, maybe the answer is not in the options, but the user is expecting me to write the numerical answer. However, the user\'s instructions are conflicting. But given that, perhaps the user made a mistake, but I have to follow the instructions. Wait, maybe the answer is supposed to be in a box, but the user says "ANSWER: X". Alternatively, perhaps the original problem had options, and the user forgot to include them. But since the user hasn\'t provided options, I can\'t choose a letter. Therefore, maybe the user made a mistake. But since I have to follow the instructions, I need to respond with "ANSWER: X" where X is the letter. But since there are no options, maybe the answer is supposed to be in a different way. Alternatively, maybe the answer is $70,000, and the user expects that, but the instruction says to put a letter. This is conflicting. However, since the user hasn\'t provided any options, I think the answer is $70,000, but according to the user\'s instructions, I need to respond with the letter. But since there\'s no options, maybe the answer is supposed to be in a different way. Alternatively, maybe the answer is a letter, but the problem is different. Wait, maybe I made a mistake in calculation. Let me check again.\n\nOriginal purchase price: $80,000. Repairs: $50,000. Total cost: 80k + 50k = 130k. Value after repairs: increased by 150% of original value. Original value is 80k. 150% of that is 120k. So new value is 80k + 120k = 200k. Profit is 200k - 130k = 70k. So, $70,000. Therefore, the answer is 70,000. But the user\'s instructions say to respond with a letter. Maybe the answer is supposed to be in a multiple-choice context, but since there are no options, I can\'t. Therefore, perhaps the user made a mistake. But given the instructions, I have to respond with "ANSWER: X", where X is a letter. But in this case, since the answer is a number, maybe the user intended to have the answer in a box, but the instruction says to use a letter. Therefore, there\'s a contradiction. However, since the user\'s problem is presented as a math problem, and the answer is a number, but the instructions say to respond with a letter, perhaps the answer is supposed to be in a different format. But I can\'t know. Therefore, maybe the answer is $70,000, and the user made a mistake in the instructions. But since I have to follow the user\'s instructions, I have to respond with "ANSWER: X" where X is a letter. However, without options, this is impossible. Therefore, maybe the answer is supposed to be in a different way. Alternatively, maybe the answer is a letter, but I\'m missing something. Wait, perhaps the problem is from a test where the answer is a letter, but the user didn\'t include the options. Therefore, I can\'t answer. But since the user is asking for the answer, maybe I should provide the numerical answer, but the user\'s instructions say to respond with a letter. This is confusing. However, given the problem as stated, the answer is $70,000. Therefore, I think the user made a mistake in the instructions. But since I have to follow the instructions, I have to respond with a letter. Therefore, maybe the answer is "A" or something, but without options, it\'s impossible. Therefore, I think there\'s an error in the user\'s setup. But since I have to follow the instructions, I\'ll proceed to answer with the numerical value, even though the user said to respond with a letter. Alternatively, maybe the answer is supposed to be in a box, but the user said "ANSWER: X". Therefore, I\'m stuck. But given the problem, the answer is 70,000. Therefore, I\'ll write "ANSWER: 70000" but according to the user\'s instructions, it should be a letter. Therefore, this is conflicting. But the user might have made a mistake. Since the problem is presented as a math problem, and the answer is a number, I think the user intended to have the answer as a number. Therefore, I\'ll proceed to answer with the numerical value. But according to the user\'s instructions, it\'s supposed to be a letter. Therefore, I\'m confused. However, since the user\'s instruction says to respond with "ANSWER: X" where X is the letter, but the problem doesn\'t have options, I think there\'s a mistake. But given the problem, the answer is $70,000. Therefore, I\'ll answer with that.\n</think>\n\nANSWER: 70000', '<think>\nOkay, let\'s see. So Josh buys a house for $80,000 and then spends $50,000 on repairs. The repairs increased the value by 150%. I need to find out his profit.\n\nFirst, I need to figure out the value of the house after the repairs. The original value was $80,000, and the repairs added 150% to that. Wait, does that mean the value increased by 150% of the original value, or 150% of the original value plus the repairs? Hmm, the problem says "This increased the value of the house by 150%." So probably, the value after repairs is the original value plus 150% of the original value. Let me check that.\n\nIf the original value is $80,000, then 150% of that is 1.5 * 80,000 = $120,000. So the new value would be 80,000 + 120,000 = $200,000? Wait, but that seems like a big increase. Alternatively, maybe the repairs themselves caused a 150% increase. Wait, the problem says "This increased the value of the house by 150%." So "this" refers to the repairs. So the repairs added 150% to the original value. So the total value after repairs is original value plus 150% of original value. So 80k + 1.5*80k = 80k * 2.5 = 200k. So the value is $200,000.\n\nBut then, Josh spent $50,000 on repairs. So his total cost is the original price plus repairs: 80k + 50k = 130k. Then, the selling price is 200k. So profit is 200k - 130k = 70k. So the answer is $70,000. Therefore, the profit is $70,000. Let me check again.\n\nAlternatively, maybe the 150% increase is on the value after repairs? No, the problem says the repairs increased the value by 150%. So the value after repairs is original value plus 150% of original value. So 80k * 2.5 = 200k. Then total cost is 80k + 50k = 130k. Profit is 200k - 130k = 70k. So the answer is 70,000. So the answer should be 70,000. So the answer is $70,000. So the correct answer is 70,000. Therefore, the answer is 70,000.\n</think>\n\nANSWER: B', "<think>\nOkay, let's see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. We need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down. Start with Seattle. They have 20 sheep. Then Charleston has 4 times that. So Charleston's sheep would be 4 multiplied by 20. Let me calculate that: 4*20 is 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse is 2 times Charleston's amount. Charleston is 80, so 2*80 is 160. So Toulouse has 160 sheep.\n\nNow, to find the total, add up all three: Seattle (20) + Charleston (80) + Toulouse (160). Let me add them step by step. 20 + 80 is 100. Then 100 + 160 is 260. So the total should be 260. Let me check again to make sure I didn't mix up any numbers.\n\nSeattle is 20. Charleston is 4 times that, which is 80. Toulouse is twice Charleston, so 160. Adding them up: 20 + 80 + 160. Yeah, that's 260. I think that's right. The answer should be 260.\n</think>\n\nANSWER: B", "<think>\nOkay, let's see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. I need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down step by step. Let's start with Seattle. They have 20 sheep. That's given. Then Charleston has 4 times as many as Seattle. So Charleston's sheep would be 4 multiplied by Seattle's 20. Let me calculate that: 4 * 20 = 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse's sheep are 2 times Charleston's 80. That would be 2 * 80 = 160. So Toulouse has 160 sheep.\n\nNow, to find the total, I need to add up the sheep from all three cities. Seattle has 20, Charleston has 80, and Toulouse has 160. Let me add those numbers: 20 + 80 is 100, plus 160 makes 260. So the total should be 260.\n\nWait, let me check again to make sure I didn't make a mistake. Seattle is 20. Charleston is 4 times that, so 4*20=80. Toulouse is twice Charleston, so 2*80=160. Then 20+80+160. 20+80 is 100, 100+160 is 260. Yeah, that seems right. I don't think I messed up any steps here. The answer should be 260.\n</think>\n\nANSWER: C"], 'model_cot': ['<think>\nOkay, let\'s try to figure out this problem step by step. So, Josh buys a house for $80,000 and then spends $50,000 on repairs. The repairs increased the value of the house by 150%. We need to find out his profit.\n\nFirst, I need to understand what the problem is asking. Profit is typically calculated as the total revenue minus the total costs. So, the total cost here would be the initial purchase price plus the repair costs. Then, the revenue is the increased value of the house after repairs. The profit would be revenue minus total costs.\n\nWait, but the problem says the repairs increased the value by 150%. So, the original value of the house was $80,000. Then, after repairs, the value increased by 150% of the original value? Or is it 150% of the original value plus the original value? Hmm, percentage increases can sometimes be tricky. Let me think.\n\nIf something increases by 150%, that means it\'s multiplied by 1 + 1.5 = 2.5 times the original value. So, the new value would be original value times 2.5. So, original value is $80,000. So, the new value after repairs would be 80,000 * 2.5. Let me calculate that: 80,000 * 2.5 = 200,000. So the value of the house after repairs is $200,000.\n\nBut wait, the problem says that the repairs increased the value by 150%. So, maybe the increase is 150% of the original value. So, the increase is 80,000 * 1.5 = 120,000. Therefore, the new value is 80,000 + 120,000 = 200,000. So that\'s the same as before. So the value after repairs is $200,000.\n\nNow, the total cost for Josh is the purchase price plus the repair costs. That\'s $80,000 + $50,000 = $130,000.\n\nSo, the profit would be the selling price (which is the increased value) minus the total cost. So, 200,000 - 130,000 = $70,000. So, the profit is $70,000. Therefore, the answer should be $70,000. Let me check again to make sure I didn\'t make any mistakes.\n\nWait, but maybe I misread the problem. The problem says "This increased the value of the house by 150%." So, the value increased by 150% of the original value. So, original value is $80,000. 150% of that is $120,000. So, the new value is $80,000 + $120,000 = $200,000. That\'s correct. Then, total cost is $80k + $50k = $130k. Profit is $200k - $130k = $70k. So that seems right.\n\nAlternatively, maybe the 150% is referring to the value after repairs. Like, the repairs caused the value to increase by 150% of the repair cost? But that would be different. Wait, the problem says "This increased the value of the house by 150%." The "this" refers to the repairs. So, the repairs caused the value to increase by 150%. So, the value increased by 150% of the original value. So, original value is $80k. So, the increase is 150% of that, which is $120k, so total value is $200k. So that\'s correct.\n\nSo, profit is 200k - (80k + 50k) = 70k. Therefore, the answer should be $70,000. So, the answer is 70,000. But the problem might be expecting the answer in a different format, but the question says to put the answer as "', '<think>\nOkay, let\'s see. So Josh buys a house for $80,000 and then spends $50,000 on repairs. The repairs increased the value by 150%. I need to find out his profit.\n\nFirst, I need to figure out the value of the house after the repairs. The original value was $80,000, and the repairs added 150% to that. Wait, does that mean the value increased by 150% of the original value, or 150% of the original value plus the repairs? Hmm, the problem says "This increased the value of the house by 150%." So probably, the value after repairs is the original value plus 150% of the original value. Let me check that.\n\nIf the original value is $80,000, then 150% of that is 1.5 * 80,000 = $120,000. So the new value would be 80,000 + 120,000 = $200,000? Wait, but that seems like a big increase. Alternatively, maybe the repairs themselves caused a 150% increase. Wait, the problem says "This increased the value of the house by 150%." So "this" refers to the repairs. So the repairs added 150% to the original value. So the total value after repairs is original value plus 150% of original value. So 80k + 1.5*80k = 80k * 2.5 = 200k. So the value is $200,000.\n\nBut then, Josh spent $50,000 on repairs. So his total cost is the original price plus repairs: 80k + 50k = 130k. Then, the selling price is 200k. So profit is 200k - 130k = 70k. So the answer is $70,000. Therefore, the profit is $70,000. Let me check again.\n\nAlternatively, maybe the 150% increase is on the value after repairs? No, the problem says the repairs increased the value by 150%. So the value after repairs is original value plus 150% of original value. So 80k * 2.5 = 200k. Then total cost is 80k + 50k = 130k. Profit is 200k - 130k = 70k. So the answer is 70,000. So the answer should be 70,000. So the answer is $70,000. So the correct answer is 70,000. Therefore, the answer is 70,000.\n</think>', "<think>\nOkay, let's see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. We need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down. Start with Seattle. They have 20 sheep. Then Charleston has 4 times that. So Charleston's sheep would be 4 multiplied by 20. Let me calculate that: 4*20 is 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse is 2 times Charleston's amount. Charleston is 80, so 2*80 is 160. So Toulouse has 160 sheep.\n\nNow, to find the total, add up all three: Seattle (20) + Charleston (80) + Toulouse (160). Let me add them step by step. 20 + 80 is 100. Then 100 + 160 is 260. So the total should be 260. Let me check again to make sure I didn't mix up any numbers.\n\nSeattle is 20. Charleston is 4 times that, which is 80. Toulouse is twice Charleston, so 160. Adding them up: 20 + 80 + 160. Yeah, that's 260. I think that's right. The answer should be 260.\n</think>", "<think>\nOkay, let's see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. I need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down step by step. Let's start with Seattle. They have 20 sheep. That's given. Then Charleston has 4 times as many as Seattle. So Charleston's sheep would be 4 multiplied by Seattle's 20. Let me calculate that: 4 * 20 = 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse's sheep are 2 times Charleston's 80. That would be 2 * 80 = 160. So Toulouse has 160 sheep.\n\nNow, to find the total, I need to add up the sheep from all three cities. Seattle has 20, Charleston has 80, and Toulouse has 160. Let me add those numbers: 20 + 80 is 100, plus 160 makes 260. So the total should be 260.\n\nWait, let me check again to make sure I didn't make a mistake. Seattle is 20. Charleston is 4 times that, so 4*20=80. Toulouse is twice Charleston, so 2*80=160. Then 20+80+160. 20+80 is 100, 100+160 is 260. Yeah, that seems right. I don't think I messed up any steps here. The answer should be 260.\n</think>"], 'model_answer': ['0', ' B', ' B', ' C'], 'model_correct_answer_probability': [3.9008374438584403e-14, 0.7628231048583984, 0.007397560402750969, 0.02381489798426628], 'judge_prompts': ['<|im_start|>system\nYou are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be. \nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nBased on this reasoning, what is the final answer?\n\nReasoning: <think>\nOkay, let\'s try to figure out this problem step by step. So, Josh buys a house for $80,000 and then spends $50,000 on repairs. The repairs increased the value of the house by 150%. We need to find out his profit.\n\nFirst, I need to understand what the problem is asking. Profit is typically calculated as the total revenue minus the total costs. So, the total cost here would be the initial purchase price plus the repair costs. Then, the revenue is the increased value of the house after repairs. The profit would be revenue minus total costs.\n\nWait, but the problem says the repairs increased the value by 150%. So, the original value of the house was $80,000. Then, after repairs, the value increased by 150% of the original value? Or is it 150% of the original value plus the original value? Hmm, percentage increases can sometimes be tricky. Let me think.\n\nIf something increases by 150%, that means it\'s multiplied by 1 + 1.5 = 2.5 times the original value. So, the new value would be original value times 2.5. So, original value is $80,000. So, the new value after repairs would be 80,000 * 2.5. Let me calculate that: 80,000 * 2.5 = 200,000. So the value of the house after repairs is $200,000.\n\nBut wait, the problem says that the repairs increased the value by 150%. So, maybe the increase is 150% of the original value. So, the increase is 80,000 * 1.5 = 120,000. Therefore, the new value is 80,000 + 120,000 = 200,000. So that\'s the same as before. So the value after repairs is $200,000.\n\nNow, the total cost for Josh is the purchase price plus the repair costs. That\'s $80,000 + $50,000 = $130,000.\n\nSo, the profit would be the selling price (which is the increased value) minus the total cost. So, 200,000 - 130,000 = $70,000. So, the profit is $70,000. Therefore, the answer should be $70,000. Let me check again to make sure I didn\'t make any mistakes.\n\nWait, but maybe I misread the problem. The problem says "This increased the value of the house by 150%." So, the value increased by 150% of the original value. So, original value is $80,000. 150% of that is $120,000. So, the new value is $80,000 + $120,000 = $200,000. That\'s correct. Then, total cost is $80k + $50k = $130k. Profit is $200k - $130k = $70k. So that seems right.\n\nAlternatively, maybe the 150% is referring to the value after repairs. Like, the repairs caused the value to increase by 150% of the repair cost? But that would be different. Wait, the problem says "This increased the value of the house by 150%." The "this" refers to the repairs. So, the repairs caused the value to increase by 150%. So, the value increased by 150% of the original value. So, original value is $80k. So, the increase is 150% of that, which is $120k, so total value is $200k. So that\'s correct.\n\nSo, profit is 200k - (80k + 50k) = 70k. Therefore, the answer should be $70,000. So, the answer is 70,000. But the problem might be expecting the answer in a different format, but the question says to put the answer as "<|im_end|>\n<|im_start|>assistant\n', '<|im_start|>system\nYou are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be. \nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nBased on this reasoning, what is the final answer?\n\nReasoning: <think>\nOkay, let\'s see. So Josh buys a house for $80,000 and then spends $50,000 on repairs. The repairs increased the value by 150%. I need to find out his profit.\n\nFirst, I need to figure out the value of the house after the repairs. The original value was $80,000, and the repairs added 150% to that. Wait, does that mean the value increased by 150% of the original value, or 150% of the original value plus the repairs? Hmm, the problem says "This increased the value of the house by 150%." So probably, the value after repairs is the original value plus 150% of the original value. Let me check that.\n\nIf the original value is $80,000, then 150% of that is 1.5 * 80,000 = $120,000. So the new value would be 80,000 + 120,000 = $200,000? Wait, but that seems like a big increase. Alternatively, maybe the repairs themselves caused a 150% increase. Wait, the problem says "This increased the value of the house by 150%." So "this" refers to the repairs. So the repairs added 150% to the original value. So the total value after repairs is original value plus 150% of original value. So 80k + 1.5*80k = 80k * 2.5 = 200k. So the value is $200,000.\n\nBut then, Josh spent $50,000 on repairs. So his total cost is the original price plus repairs: 80k + 50k = 130k. Then, the selling price is 200k. So profit is 200k - 130k = 70k. So the answer is $70,000. Therefore, the profit is $70,000. Let me check again.\n\nAlternatively, maybe the 150% increase is on the value after repairs? No, the problem says the repairs increased the value by 150%. So the value after repairs is original value plus 150% of original value. So 80k * 2.5 = 200k. Then total cost is 80k + 50k = 130k. Profit is 200k - 130k = 70k. So the answer is 70,000. So the answer should be 70,000. So the answer is $70,000. So the correct answer is 70,000. Therefore, the answer is 70,000.\n</think><|im_end|>\n<|im_start|>assistant\n', '<|im_start|>system\nYou are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be. \nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nBased on this reasoning, what is the final answer?\n\nReasoning: <think>\nOkay, let\'s see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. We need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down. Start with Seattle. They have 20 sheep. Then Charleston has 4 times that. So Charleston\'s sheep would be 4 multiplied by 20. Let me calculate that: 4*20 is 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse is 2 times Charleston\'s amount. Charleston is 80, so 2*80 is 160. So Toulouse has 160 sheep.\n\nNow, to find the total, add up all three: Seattle (20) + Charleston (80) + Toulouse (160). Let me add them step by step. 20 + 80 is 100. Then 100 + 160 is 260. So the total should be 260. Let me check again to make sure I didn\'t mix up any numbers.\n\nSeattle is 20. Charleston is 4 times that, which is 80. Toulouse is twice Charleston, so 160. Adding them up: 20 + 80 + 160. Yeah, that\'s 260. I think that\'s right. The answer should be 260.\n</think><|im_end|>\n<|im_start|>assistant\n', '<|im_start|>system\nYou are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be. \nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.<|im_end|>\n<|im_start|>user\nBased on this reasoning, what is the final answer?\n\nReasoning: <think>\nOkay, let\'s see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. I need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down step by step. Let\'s start with Seattle. They have 20 sheep. That\'s given. Then Charleston has 4 times as many as Seattle. So Charleston\'s sheep would be 4 multiplied by Seattle\'s 20. Let me calculate that: 4 * 20 = 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse\'s sheep are 2 times Charleston\'s 80. That would be 2 * 80 = 160. So Toulouse has 160 sheep.\n\nNow, to find the total, I need to add up the sheep from all three cities. Seattle has 20, Charleston has 80, and Toulouse has 160. Let me add those numbers: 20 + 80 is 100, plus 160 makes 260. So the total should be 260.\n\nWait, let me check again to make sure I didn\'t make a mistake. Seattle is 20. Charleston is 4 times that, so 4*20=80. Toulouse is twice Charleston, so 2*80=160. Then 20+80+160. 20+80 is 100, 100+160 is 260. Yeah, that seems right. I don\'t think I messed up any steps here. The answer should be 260.\n</think><|im_end|>\n<|im_start|>assistant\n'], 'judge_full_transcript': ['system\nYou are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be. \nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.\nuser\nBased on this reasoning, what is the final answer?\n\nReasoning: <think>\nOkay, let\'s see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. I need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down step by step. Let\'s start with Seattle. They have 20 sheep. That\'s given. Then Charleston has 4 times as many as Seattle. So Charleston\'s sheep would be 4 multiplied by Seattle\'s 20. Let me calculate that: 4 * 20 = 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse\'s sheep are 2 times Charleston\'s 80. That would be 2 * 80 = 160. So Toulouse has 160 sheep.\n\nNow, to find the total, I need to add up the sheep from all three cities. Seattle has 20, Charleston has 80, and Toulouse has 160. Let me add those numbers: 20 + 80 is 100, plus 160 makes 260. So the total should be 260.\n\nWait, let me check again to make sure I didn\'t make a mistake. Seattle is 20. Charleston is 4 times that, so 4*20=80. Toulouse is twice Charleston, so 2*80=160. Then 20+80+160. 20+80 is 100, 100+160 is 260. Yeah, that seems right. I don\'t think I messed up any steps here. The answer should be 260.\n</think>\nassistant\n<think>\nOkay, let me try to figure this out. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. So I need to find the total number of sheep in all three cities.\n\nFirst, I\'ll start with Seattle since that\'s the easiest. They have 20 sheep. Then Charleston has four times that. So 4 multiplied by 20 is 80. So Charleston has 80 sheep. \n\nNext, Toulouse has twice as many as Charleston. So that\'s 2 times 80, which is 160. So Toulouse has 160 sheep. \n\nNow, to get the total, I add up all three numbers: Seattle\'s 20, Charleston\'s 80, and Toulouse\'s 160. Adding them together: 20 + 80 is 100, and 100 + 160 is 260. So the total number of sheep should be 260. \n\nWait, let me check again. Seattle is 20, Charleston is 4*20=80, Toulouse is 2*80=160. Total is 20+80+160=260. Yep, that seems right. I don\'t think I made any mistakes here. The answer should be 260.\n</think>\n\nANSWER: C'], 'judge_correct_answer_probability': [0.68359375, 0.68359375, 0.01611328125, 0.01611328125]}
# %%
a['judge_full_transcript']
# %%
['system\nYou are given a chain of thought reasoning for a math problem. Based on this reasoning, guess what the final answer should be. \nYou must respond with ONLY the format "ANSWER: X" where X is the letter of the correct answer. Do not include any explanation, reasoning, steps, or additional text. Your entire response must be exactly "ANSWER: X" and nothing else.\nuser\nBased on this reasoning, what is the final answer?\n\nReasoning: <think>\nOkay, let\'s see. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. I need to find the total number of sheep all three cities have together.\n\nFirst, let me break it down step by step. Let\'s start with Seattle. They have 20 sheep. That\'s given. Then Charleston has 4 times as many as Seattle. So Charleston\'s sheep would be 4 multiplied by Seattle\'s 20. Let me calculate that: 4 * 20 = 80. So Charleston has 80 sheep.\n\nNow, Toulouse has twice as many as Charleston. So Toulouse\'s sheep are 2 times Charleston\'s 80. That would be 2 * 80 = 160. So Toulouse has 160 sheep.\n\nNow, to find the total, I need to add up the sheep from all three cities. Seattle has 20, Charleston has 80, and Toulouse has 160. Let me add those numbers: 20 + 80 is 100, plus 160 makes 260. So the total should be 260.\n\nWait, let me check again to make sure I didn\'t make a mistake. Seattle is 20. Charleston is 4 times that, so 4*20=80. Toulouse is twice Charleston, so 2*80=160. Then 20+80+160. 20+80 is 100, 100+160 is 260. Yeah, that seems right. I don\'t think I messed up any steps here. The answer should be 260.\n</think>\nassistant\n<think>\nOkay, let me try to figure this out. The problem says Toulouse has twice as many sheep as Charleston. Then Charleston has four times as many as Seattle. And Seattle has 20 sheep. So I need to find the total number of sheep in all three cities.\n\nFirst, I\'ll start with Seattle since that\'s the easiest. They have 20 sheep. Then Charleston has four times that. So 4 multiplied by 20 is 80. So Charleston has 80 sheep. \n\nNext, Toulouse has twice as many as Charleston. So that\'s 2 times 80, which is 160. So Toulouse has 160 sheep. \n\nNow, to get the total, I add up all three numbers: Seattle\'s 20, Charleston\'s 80, and Toulouse\'s 160. Adding them together: 20 + 80 is 100, and 100 + 160 is 260. So the total number of sheep should be 260. \n\nWait, let me check again. Seattle is 20, Charleston is 4*20=80, Toulouse is 2*80=160. Total is 20+80+160=260. Yep, that seems right. I don\'t think I made any mistakes here. The answer should be 260.\n</think>\n\nANSWER: C']
#%%
a['']