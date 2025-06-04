"""Dataset utilities to convert MCQ rows into ChatML prompts for Qwen‑3.

This version uses **Jinja 2** to build the prompt so the structure is
clear and easily editable.  It follows the ChatML format we discussed,
including the `<think>` block for chain‑of‑thought.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from jinja2 import Template

import torch
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Jinja template blocks
# ---------------------------------------------------------------------------

_SYSTEM_BLOCK = (
    "<|im_start|>system\n"
    "You are a helpful assistant specialised in master‑level STEM.\n"
    "<|im_end|>\n"
)

_USER_TMPL = Template(
    """<|im_start|>user
The following is a multiple choice question (with answers) about knowledge and skills in advanced master‑level STEM courses.

Question: {{ question }}
Choices:
{{ choices_block }}
<|im_end|>
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

# MMLU-specific template
_MMLU_USER_TMPL = Template(
    """<|im_start|>user
The following are multiple choice questions (with answers) about knowledge and skills in advanced master‑level STEM courses.
Just answer with A, B, C, or D.

{{ question }}
{{ choices_block }}
Answer:<|im_end|>
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

_OPEN_ANSWER_TMPL = Template(
    """<|im_start|>user
The following is a question about knowledge and skills in advanced master‑level STEM courses.

Question: {{ question }}

<|im_end|>
""",
    trim_blocks=True,
    lstrip_blocks=True,
)

_ASSISTANT_START = "<|im_start|>assistant\n"

_ASSISTANT_BODY_TMPL = Template(
    """<think>
</think>
{{ answer_text }}""",
    trim_blocks=True,
    lstrip_blocks=True,
)

_ASSISTANT_END = "\n<|im_end|>"

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_prompt_section(question: str, choices_block: str, use_mmlu: bool = False) -> str:
    """Return the system + user blocks ready for concatenation."""
    if use_mmlu:
        return _SYSTEM_BLOCK + _MMLU_USER_TMPL.render(
            question=question, choices_block=choices_block
        )
    return _SYSTEM_BLOCK + _USER_TMPL.render(
        question=question, choices_block=choices_block
    )


def process_mcq_dataset(
    row: Dict[str, str | Sequence[str]],
    *,
    tokenizer=None,
    use_mmlu: bool = False,
) -> Dict[str, str | int | None]:
    """Convert one *row* into the structure required by the training pipeline.

    Parameters
    ----------
    row : dict
        Must contain ``question``, ``choices``, ``explanation`` and ``answer_text``.
    tokenizer : Pre‑trained tokenizer (optional)
        If supplied, we compute ``prompt_len`` (number of tokens in *prompt*)
        so the collator can create an attention mask faster.
    use_mmlu : bool
        If True, uses MMLU-style formatting with letter-only answers.

    Returns
    -------
    dict
        ``{"prompt", "text", "prompt_len"``
    """

    # --- 1. Choices block ---------------------------------------------------
    choices_val = row["choices"]
    if isinstance(choices_val, (list, tuple)):
        choices_block = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(choices_val)
        )
    else:
        choices_block = str(choices_val)

    # --- 2. Prompt (system+user+assistant header) ---------------------------
    prompt = build_prompt_section(row["question"], choices_block, use_mmlu=use_mmlu)
    
    if not use_mmlu:
        prompt += _ASSISTANT_START

    # --- 3. Assistant body --------------------------------------------------
    if use_mmlu:
        # For MMLU, just use the letter answer
        answer_text = chr(65 + row["answer_index"])
    else:
        assistant_body = _ASSISTANT_BODY_TMPL.render(
            answer_text=row["answer_text"],
        )
        answer_text = assistant_body

    # --- 4. Full text -------------------------------------------------------
    if use_mmlu:
        text = prompt + f" {answer_text}"
    else:
        text = prompt + answer_text + _ASSISTANT_END

    # --- 5. Prompt length (optional) ---------------------------------------
    if tokenizer is not None:
        prompt_len = len(tokenizer(prompt)["input_ids"])
    else:
        prompt_len = None

    return {
        "prompt": prompt,
        "text": text,
        "prompt_len": prompt_len,
    }


def process_open_answer_dataset(
    row: Dict[str, str],
    *,
    tokenizer=None,
) -> Dict[str, str | int | None]:
    """Convert one *row* into the structure required by the training pipeline for open answer questions.

    Parameters
    ----------
    row : dict
        Must contain ``question``, ``answer``, and optionally ``explanation``.
    tokenizer : Pre-trained tokenizer (optional)
        If supplied, we compute ``prompt_len`` (number of tokens in *prompt*)
        so the collator can create an attention mask faster.

    Returns
    -------
    dict
        ``{"prompt", "text", "prompt_len"}``
    """

    # --- 1. Prompt (system+user+assistant header) ---------------------------
    prompt = (
        _SYSTEM_BLOCK
        + _OPEN_ANSWER_TMPL.render(question=row["question"])
        + _ASSISTANT_START
    )

    # --- 2. Assistant body --------------------------------------------------
    assistant_body = _ASSISTANT_BODY_TMPL.render(
        # explanation=row.get("explanation", ""),  # Optional explanation
        answer_text=row["answer"],
    )

    # --- 3. Full text -------------------------------------------------------
    text = prompt + assistant_body + _ASSISTANT_END

    # --- 4. Prompt length (optional) ---------------------------------------
    if tokenizer is not None:
        prompt_len = len(tokenizer(prompt)["input_ids"])
    else:
        prompt_len = None

    return {
        "prompt": prompt,
        "text": text,
        "prompt_len": prompt_len,
    }


def tokenize_func(ex, tokenizer):
    # Tokenize with truncation - using 8192 tokens to accommodate long explanations
    # Qwen models support up to 8192 tokens
    tok = tokenizer(ex["text"], truncation=True, max_length=8_192)
    tok["prompt_len"] = ex["prompt_len"]  # keep for masking later
    return tok


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        try:
            # Handle single sample case (batch_size=1)
            if len(batch) == 1:
                b = batch[0]
                input_ids = torch.as_tensor(b["input_ids"], dtype=torch.long).unsqueeze(
                    0
                )  # Add batch dimension
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
                # Mask out prompt part if prompt_len exists
                if "prompt_len" in b and b["prompt_len"] is not None:
                    labels[0, : b["prompt_len"]] = -100  # ignore prompt tokens in loss
            else:
                # Convert list of tokenized samples to padded tensor batch
                input_ids_list = [
                    torch.as_tensor(b["input_ids"], dtype=torch.long) for b in batch
                ]
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids_list,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )

                # Attention mask: 1 for real tokens, 0 for pad
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

                # Create labels (copy of input_ids)
                labels = input_ids.clone()

                # Mask out prompt part if prompt_len exists
                for i, b in enumerate(batch):
                    if "prompt_len" in b and b["prompt_len"] is not None:
                        prompt_len = b["prompt_len"]  # length of prompt in tokens
                        labels[i, :prompt_len] = -100  # ignore prompt tokens in loss
                
                # Mask padding tokens in labels
                labels[input_ids == self.tokenizer.pad_token_id] = -100

            # Debugging: Check if we have valid labels for loss computation
            valid_labels = (labels != -100).sum().item()
            if valid_labels == 0:
                print(f"WARNING: No valid labels found for loss computation!")
                print(f"Input shape: {input_ids.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Prompt lengths: {[b.get('prompt_len', 'None') for b in batch]}")

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        except Exception as e:
            # Log the error and batch information for debugging
            print(f"Error in DataCollator: {str(e)}")
            print(f"Batch keys: {[b.keys() for b in batch]}")
            raise
