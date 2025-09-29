# Refactoring Summary: Multi-Choice Utils & RL Setup

## Changes Made

### 1. Created `src/multichoice_utils.py`
A new module containing reusable multi-choice helper functions:

- **`load_gsm8k_mc_sample(index)`** - Load sample from guipenedo/gsm8k-mc dataset
- **`format_multiple_choice_question(sample)`** - Format question with A/B/C/D choices
- **`get_answer_token_candidates(tokenizer, answer_choice)`** - Get token IDs for answer choices
- **`extract_answer_probability(generated_tokens, all_probabilities, tokenizer, correct_answer)`** - Extract probability for correct answer
- **`extract_answer(response)`** - Parse "ANSWER: X" format
- **`extract_cot(response)`** - Extract chain of thought
- **`answers_match(predicted, correct)`** - Check if answers match
- **`calculate_reward(prob_correct_model, prob_correct_judge)`** - **NEW**: Reward function for RL training

### 2. Created `rl_training.py`
RL training script with structure for GRPO training:

**Key Features:**
- Unsloth integration for optimized training
- LoRA configuration for efficient fine-tuning
- 4-bit quantization support
- Gradient checkpointing
- Judge model loading (frozen)
- Reward calculation: `R = P(correct|model) × (1 - P(correct|judge))`
- Training loop structure (GRPO implementation placeholder)

**Functions:**
- `setup_lora_config()` - Configure LoRA parameters
- `load_model_with_unsloth()` - Load model with Unsloth optimizations
- `load_judge_model()` - Load frozen judge for evaluation
- `generate_with_probabilities()` - Generate response with probability tracking
- `compute_reward()` - Calculate reward for a single question
- `train_grpo()` - Main training loop

### 3. Created `test_single_forward_pass.py`
Simple test script to verify a single forward pass:

**What it does:**
1. Loads a single question from GSM8K-MC
2. Runs model to generate answer with CoT
3. Runs judge to guess answer from CoT
4. Calculates probabilities: `P(correct|model)` and `P(correct|judge)`
5. Computes reward: `R = P(correct|model) × (1 - P(correct|judge))`
6. Prints detailed results and interpretation

**Usage:**
```bash
python test_single_forward_pass.py
```

### 4. Updated `minimal_forward_pass.py`
Refactored to use `multichoice_utils`:

**Changes:**
- Removed duplicate function definitions
- Added import from `multichoice_utils`
- Cleaner, more maintainable code
- All multi-choice logic now in one place

## Reward Function

The reward function encourages:
- **High probability of correct answer**: `P(correct|model)` should be high
- **Judge being fooled**: `P(correct|judge)` should be low

Formula: `R = P(correct|model) × (1 - P(correct|judge))`

**Scenarios:**
- **IDEAL**: Model correct, judge fooled → HIGH reward
- **OKAY**: Model correct, judge correct → MEDIUM reward
- **BAD**: Model wrong, judge wrong → LOW reward
- **WORST**: Model wrong, judge correct → VERY LOW reward

## Next Steps for RL Integration

### To implement full GRPO training:

1. **Install dependencies:**
   ```bash
   pip install unsloth trl peft bitsandbytes
   ```

2. **Implement GRPO algorithm:**
   - The current `rl_training.py` has the structure but needs the actual GRPO update step
   - Can use TRL's `PPOTrainer` as a base or implement custom GRPO
   - Need to compute policy gradients and update model weights

3. **Configure training:**
   - Adjust hyperparameters in `setup_ppo_config()`
   - Set batch size, learning rate, etc.
   - Configure logging (wandb/tensorboard)

4. **Run training:**
   ```bash
   python rl_training.py
   ```

5. **Test:**
   ```bash
   python test_single_forward_pass.py
   ```

## Files Structure

```
model-organisms-cot-pathologies/
├── src/
│   ├── multichoice_utils.py          [NEW] Multi-choice helper functions
│   ├── dataset_loader.py              [Existing] Dataset utilities
│   └── gsm8k_evaluator.py             [Existing] Evaluation utilities
├── minimal_forward_pass.py            [Updated] Uses multichoice_utils
├── rl_training.py                     [NEW] RL training with GRPO
├── test_single_forward_pass.py        [NEW] Test single forward pass
└── REFACTORING_SUMMARY.md             [NEW] This file
```

## Technologies Used

- **Unsloth**: Optimized model loading and training
- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **TRL**: Transformer Reinforcement Learning library
- **GRPO**: Group Relative Policy Optimization (to be implemented)
- **4-bit quantization**: Memory-efficient training
- **Gradient checkpointing**: Reduced memory usage