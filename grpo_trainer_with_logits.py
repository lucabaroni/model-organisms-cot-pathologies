"""
Patched GRPOTrainer that passes logits to the reward function.

This is a minimal modification of TRL's GRPOTrainer to pass the completion logits
to the reward function, avoiding the need to re-generate or re-compute logits.
"""

import torch
from typing import Any, Union, Optional
from trl import GRPOTrainer as TRLGRPOTrainer


class GRPOTrainerWithLogits(TRLGRPOTrainer):
    """
    GRPOTrainer that passes completion logits to the reward function.

    This allows reward functions to access the model's probability distribution
    over tokens without needing to re-generate or re-compute a forward pass.

    The reward function will receive additional kwargs:
    - completion_logits: List of tensors, shape (completion_length, vocab_size) for each completion
    - completion_ids: List of token IDs (already provided by base class)
    """

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Override to store logits for passing to reward function.

        This method generates completions and computes their logits, then stores
        the logits so they can be passed to the reward function.
        """
        # Call parent implementation to do all the generation
        outputs = super()._generate_and_score_completions(inputs)

        # The parent already computed logits for policy gradient, but didn't save them
        # We need to re-compute them once more to get the full logits (not just log probs)
        # This is still more efficient than re-generating in the reward function

        # Extract what we need from outputs
        prompt_completion_ids = outputs["prompt_completion_ids"]
        attention_mask = outputs["attention_mask"]
        completion_ids = outputs["completion_ids"]

        # Get the logits for completions
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if self.model.training else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # Compute logits (similar to what _get_per_token_logps_and_entropies does)
            all_logits = []
            for start in range(0, prompt_completion_ids.size(0), batch_size):
                input_ids_batch = prompt_completion_ids[start : start + batch_size]
                attention_mask_batch = attention_mask[start : start + batch_size]

                model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
                model_inputs["use_cache"] = False

                if "logits_to_keep" in self.model_kwarg_keys:
                    model_inputs["logits_to_keep"] = logits_to_keep + 1

                batch_logits = self.model(**model_inputs).logits
                # Exclude the last value: it corresponds to the next token pred
                batch_logits = batch_logits[:, :-1, :]  # (B, L-1, H)
                # Only keep the last logits_to_keep
                batch_logits = batch_logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
                # Divide by temperature (same as GRPO does)
                batch_logits = batch_logits / self.temperature

                all_logits.append(batch_logits)

            completion_logits_tensor = torch.cat(all_logits, dim=0)  # (total_completions, completion_len, vocab_size)

        # Store logits as an instance variable so _calculate_rewards can access them
        # We can't add it to outputs dict because that gets converted to a list later
        self._cached_completion_logits = completion_logits_tensor

        return outputs

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """
        Override to pass logits to the reward function.

        This method extracts the logits stored during generation and passes them
        to the reward function via kwargs.
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Get the completion logits if available (from our modified _generate_and_score_completions)
        # They're stored as an instance variable
        completion_logits = getattr(self, '_cached_completion_logits', None)

        # Convert logits tensor to list of tensors (one per completion)
        if completion_logits is not None:
            # completion_logits shape: (num_completions, max_completion_len, vocab_size)
            completion_logits_list = [completion_logits[i] for i in range(len(prompts))]
        else:
            completion_logits_list = None

        # Repeat all input columns (but "prompt", "completion", "completion_ids")
        # inputs is a list of dicts, one per sample
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # Add completion_logits to kwargs if available
        if completion_logits_list is not None:
            reward_kwargs["completion_logits"] = completion_logits_list

        # This allows for dynamic reward shaping based on training progress
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            # For model-based reward functions
            if isinstance(reward_func, torch.nn.Module):
                # Use the parent's implementation
                if hasattr(inputs, "__getitem__") and not isinstance(inputs, dict):
                    # inputs is a list
                    from trl.data_utils import is_conversational, apply_chat_template

                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super(TRLGRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Custom reward function - pass the logits!
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs
                )
                # Convert None values to NaN
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
                if key != "trainer_state" and key != "completion_logits"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            import logging
            logger = logging.get_logger(__name__)
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function
        from accelerate.utils import gather
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func