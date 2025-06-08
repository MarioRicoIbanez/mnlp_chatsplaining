"""Evaluation utilities for model assessment."""

import logging
import torch
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils.dataset_utils import (
    _SYSTEM_BLOCK,
    _OPEN_ANSWER_TMPL,
    _ASSISTANT_START,
    process_mcq_dataset,
)

logger = logging.getLogger(__name__)


def load_mcqa_test_data(
    tokenizer: PreTrainedTokenizer, num_samples: int = 6
) -> Dataset:
    """
    Load test samples from the MCQA dataset.

    Args:
        tokenizer: The tokenizer to use for processing
        num_samples: Number of test samples to load

    Returns:
        Dataset containing processed test samples
    """
    # Load test split from MCQA dataset
    test_dataset = load_dataset("jonlecumberri/mcqa_merged", split="test")

    # Take specified number of samples
    test_samples = test_dataset.select(range(min(num_samples, len(test_dataset))))

    # Process samples using the same function as training data
    processed_samples = test_samples.map(
        process_mcq_dataset,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=test_samples.column_names,
    )

    return processed_samples


def load_openqa_test_data(num_samples_per_dataset: int = 3) -> Dataset:
    """
    Load test samples from OpenCode and OpenMath datasets.

    Args:
        num_samples_per_dataset: Number of samples to load from each dataset

    Returns:
        Dataset containing test samples from both datasets
    """
    # Load test splits from both datasets
    opencode_test = load_dataset("RikoteMaster/OpenCodeTreated", split="train")
    openmath_test = load_dataset("RikoteMaster/OpenMathTreated", split="train")

    # Take specified number of samples from each
    opencode_samples = opencode_test.select(
        range(min(num_samples_per_dataset, len(opencode_test)))
    )
    openmath_samples = openmath_test.select(
        range(min(num_samples_per_dataset, len(openmath_test)))
    )

    # Add source information
    opencode_samples = opencode_samples.map(lambda x: {"source": "OpenCode", **x})
    openmath_samples = openmath_samples.map(lambda x: {"source": "OpenMath", **x})

    # Combine samples
    combined_samples = Dataset.from_dict(
        {
            "question": opencode_samples["question"] + openmath_samples["question"],
            "source": opencode_samples["source"] + openmath_samples["source"],
        }
    )

    return combined_samples


def evaluate_model_on_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_samples: Dataset,
    max_new_tokens: int = 2000,
    temperature: float = 0.7,
) -> Dict[str, float]:
    """
    Evaluate model on test samples and return accuracy metrics.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_samples: Dataset containing test samples
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation

    Returns:
        Dict containing accuracy metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Model predictions vs Ground Truth")
    print("=" * 80)

    correct_predictions = 0
    total_predictions = 0

    for i, sample in enumerate(test_samples):
        print(f"\n--- SAMPLE {i+1} ---")
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Choices: {sample.get('choices', 'N/A')}")
        print(f"Ground Truth Answer: {sample.get('answer_text', 'N/A')}")

        # Use only the prompt (without the answer) for generation
        prompt_only = sample["prompt"]

        # Tokenize and generate
        inputs = tokenizer(prompt_only, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (excluding the input prompt)
        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        print(f"\nModel Generated:")
        print(f"'{generated_text}'")

        # Simple evaluation: check if the correct answer letter appears in generated text
        ground_truth = sample.get("answer_text", "")
        choices = sample.get("choices", [])

        if isinstance(choices, list) and ground_truth in choices:
            correct_answer_index = choices.index(ground_truth)
            correct_letter = chr(65 + correct_answer_index)  # A, B, C, D

            # Check if the correct letter appears in the generated response
            is_correct = correct_letter in generated_text.upper()
            correct_predictions += int(is_correct)
            total_predictions += 1

            print(f"Expected Answer: {correct_letter}. {ground_truth}")
            print(f"Prediction: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        else:
            print(f"Could not evaluate this sample")

        print("-" * 60)

    # Calculate and return metrics
    metrics = {}
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        metrics["accuracy"] = accuracy
        metrics["correct_predictions"] = correct_predictions
        metrics["total_predictions"] = total_predictions
        print(
            f"\nOVERALL ACCURACY: {correct_predictions}/{total_predictions} = {accuracy:.1f}%"
        )
    else:
        print(f"\nCould not compute accuracy")
        metrics["accuracy"] = 0.0
        metrics["correct_predictions"] = 0
        metrics["total_predictions"] = 0

    return metrics


def evaluate_sciq(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_samples: Dataset,
    max_new_tokens: int = 2000,
    temperature: float = 0.7,
) -> Dict[str, float]:
    """
    Evaluate model on SciQ dataset samples and return accuracy metrics.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_samples: Dataset containing test samples
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation

    Returns:
        Dict containing accuracy metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATION: Model predictions vs Ground Truth (SciQ)")
    print("=" * 80)

    correct_predictions = 0
    total_predictions = 0

    for i, sample in enumerate(test_samples):
        print(f"\n--- SAMPLE {i+1} ---")
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Choices: {sample.get('choices', 'N/A')}")
        print(f"Ground Truth Answer: {sample.get('answer_text', 'N/A')}")

        # Use only the prompt (without the answer) for generation
        prompt_only = sample["prompt"]

        # Tokenize and generate
        inputs = tokenizer(prompt_only, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part (excluding the input prompt)
        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        print(f"\nModel Generated:")
        print(f"'{generated_text}'")

        # Simple evaluation: check if the correct answer letter appears in generated text
        ground_truth = sample.get("answer_text", "")
        choices = sample.get("choices", [])

        if isinstance(choices, list) and ground_truth in choices:
            correct_answer_index = choices.index(ground_truth)
            correct_letter = chr(65 + correct_answer_index)  # A, B, C, D

            # Check if the correct letter appears in the generated response
            is_correct = correct_letter in generated_text.upper()
            correct_predictions += int(is_correct)
            total_predictions += 1

            print(f"Expected Answer: {correct_letter}. {ground_truth}")
            print(f"Prediction: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        else:
            print(f"Could not evaluate this sample")

        print("-" * 60)

    # Calculate and return metrics
    metrics = {}
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        metrics["accuracy"] = accuracy
        metrics["correct_predictions"] = correct_predictions
        metrics["total_predictions"] = total_predictions
        print(
            f"\nOVERALL ACCURACY: {correct_predictions}/{total_predictions} = {accuracy:.1f}%"
        )
    else:
        print(f"\nCould not compute accuracy")
        metrics["accuracy"] = 0.0
        metrics["correct_predictions"] = 0
        metrics["total_predictions"] = 0

    return metrics


def evaluate_openqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_samples: Dataset,
    max_new_tokens: int = 2000,
    temperature: float = 0.7,
) -> None:
    """
    Generate responses for OpenQA (OpenCode/OpenMath) samples.
    This is a qualitative evaluation that just shows the model's responses.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_samples: Dataset containing test samples
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation
    """
    print("\n" + "=" * 80)
    print("MODEL RESPONSES: OpenQA Samples")
    print("=" * 80)

    for i, sample in enumerate(test_samples):
        print(f"\n--- SAMPLE {i+1} ({sample.get('source', 'Unknown')}) ---")
        print(f"Question: {sample.get('question', 'N/A')}")

        # Build prompt using the same templates as in dataset_utils.py
        prompt = (
            _SYSTEM_BLOCK
            + _OPEN_ANSWER_TMPL.render(question=sample["question"])
            + _ASSISTANT_START
        )

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False
        ).strip()

        print("\nModel Response:")
        print(f"'{generated_text}'")

        print("-" * 60)
