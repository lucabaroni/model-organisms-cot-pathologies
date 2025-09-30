# GRPO Logits Optimization Summary

## Problem

When computing the reward `R = P(correct|model) × (1 - P(correct|judge))`, the original code was inefficient:

1. **GRPO generates completions** and computes logits internally
2. GRPO **discards the logits** after using them for policy gradients
3. **Reward function re-generates** the same completions to get probabilities
4. This **doubles the computation time** for the model forward passes

## Solution

We created a minimal patch to GRPO that passes logits to the reward function, eliminating the need to re-generate.

### Files Created/Modified

1. **`grpo_trainer_with_logits.py`** - Patched GRPOTrainer
   - Extends TRL's GRPOTrainer
   - Overrides `_generate_and_score_completions()` to store logits
   - Overrides `_calculate_rewards()` to pass logits to reward function via `completion_logits` kwarg

2. **`src/multichoice_utils.py`** - Added optimized function
   - `extract_answer_probability_from_logits()` - Extracts probability from logits
   - Only looks at the position where the answer token appears (not all positions)
   - Avoids any model forward passes

3. **`test_rl_training_optimized.py`** - Updated test
   - Uses `GRPOTrainerWithLogits` instead of standard `GRPOTrainer`
   - Reward function checks for `completion_logits` kwarg
   - Falls back to re-generation if logits not available

## How It Works

### Before (Inefficient)
```
1. GRPO generates: model.generate(prompt) → completion + logits
2. GRPO uses logits for policy gradient, then discards them
3. Reward function: model.generate(prompt) again → completion + logits  ❌ DUPLICATE!
4. Reward function: judge.generate(cot) → judge_response + logits
5. Compute reward from probabilities
```

### After (Optimized)
```
1. GRPO generates: model.generate(prompt) → completion + logits
2. GRPO saves logits and passes to reward function ✓
3. Reward function: extract P(correct) directly from logits  ✓ NO REGENERATION!
4. Reward function: judge.generate(cot) → judge_response + logits
5. Compute reward from probabilities
```

## Key Optimization Details

### Position-based Probability Extraction

The `extract_answer_probability_from_logits()` function is optimized to:
1. Find which position in the completion has an answer token (A/B/C/D)
2. Extract logits **only at that position**: `logits[answer_pos]` → `[vocab_size]`
3. Convert to probabilities: `softmax(logits[answer_pos])` → `[vocab_size]`
4. Return `P(correct_answer_token)`

**Why this matters:**
- Completion length: 100-500 tokens
- Vocab size: ~50,000 tokens
- We only need probability at 1 position, not all positions!

### What Gets Passed to Reward Function

The reward function now receives:
```python
def reward_fn(prompts, completions, completion_ids, completion_logits, **kwargs):
    """
    completion_logits: List of tensors, one per completion
                      Each tensor has shape [completion_length, vocab_size]
    completion_ids: List of token ID lists (already provided by base GRPO)
    """
```

## Performance Impact

**Time saved per reward computation:**
- Before: 2 model generations (model + judge)
- After: 1 model generation (judge only) + 1 softmax operation
- **Speedup: ~2x for reward computation**

**GPU memory:**
- Slightly higher during generation (need to store logits)
- But no additional generation passes, so net positive

## Usage

Replace standard GRPOTrainer with the patched version:

```python
# Old
from trl import GRPOTrainer
trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, ...)

# New
from grpo_trainer_with_logits import GRPOTrainerWithLogits
trainer = GRPOTrainerWithLogits(model=model, reward_funcs=reward_fn, ...)
```

Update reward function to use logits:

```python
def reward_fn(prompts, completions, completion_ids=None, completion_logits=None, **kwargs):
    # Check if optimized logits available
    if completion_logits is not None and completion_ids is not None:
        prob = extract_answer_probability_from_logits(
            completion_logits[i], completion_ids[i], tokenizer, correct_answer
        )
    else:
        # Fallback to re-generation
        prob = extract_probability_by_regenerating(...)
```

## Testing

Run the optimized test:
```bash
python test_rl_training_optimized.py
```

This will:
1. Show that logits are being passed to the reward function
2. Demonstrate the optimized probability extraction
3. Train for 1 epoch on 2 samples to verify everything works

## Notes

- The patched trainer is fully compatible with standard GRPO
- Fallback behavior ensures compatibility if logits aren't available
- Judge generation still requires a forward pass (no way around this)
- Model probability extraction is now zero-cost (just indexing logits)