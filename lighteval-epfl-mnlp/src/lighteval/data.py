# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
import logging
import math
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch.utils.data import Dataset


if version.parse(torch.__version__) >= version.parse("2.5.0"):
    from torch.utils.data.distributed import DistributedSampler, _T_co
else:
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data.distributed import T_co as _T_co

from lighteval.tasks.requests import (
    GreedyUntilRequest,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
    LoglikelihoodSingleTokenRequest,
    DPOInferenceRequest,
    Request,
)


logger = logging.getLogger(__name__)

class DefaultDataset(Dataset):
    def __init__(self, requests: list):
        self.requests = requests

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        return self.requests[index]

class DynamicBatchDataset(Dataset):
    def __init__(
        self,
        requests: list,
        num_dataset_splits: int,
    ):
        """
        This dataset class uses dynamic batching to speed up the generation.
        Each request is sorted by the length of the prompt + the length of the
        continuation. Then, the dataset is split into num_dataset_splits splits.
        The first split will contain the longest requests, the second split will
        contain the second longest requests, etc. This allows us to use dynamic
        batching by starting with a small batch size and doubling it for each
        split. This is much faster than using a fixed batch size for the whole
        dataset.

        Args:
            requests (List): A list of requests.
            num_dataset_splits (int): The number of dataset splits.
        """
        # We make sure the requests contain the tokenized versions of their values
        if any(r.tokenized_context is None for r in requests):
            raise ValueError("You passed a request for which tokenization had not happened yet.")

        # sort the requests using the collate function and save the original order
        enumerated_requests = list(enumerate(requests))
        sorted_enumerated_requests = sorted(enumerated_requests, key=lambda x: self._sorting_criteria(x[1]))

        self.sorted_data = [x[1] for x in sorted_enumerated_requests]
        self.original_order = [x[0] for x in sorted_enumerated_requests]

        self.total_size = len(self.sorted_data)

        self.num_dataset_splits, self.splits = self.init_split_limits(num_dataset_splits)

        self.split_start, self.split_end = self.splits[0]

    def init_split_limits(self, num_dataset_splits):
        if num_dataset_splits >= self.total_size:
            logger.warning(
                f"num_dataset_splits ({num_dataset_splits}) >= total_size ({self.total_size}), setting num_dataset_splits to 1"
            )
            num_dataset_splits = 1

        split_size = math.ceil(self.total_size / num_dataset_splits)
        splits_indices = [
            (ix * split_size, min((ix + 1) * split_size, self.total_size)) for ix in range(num_dataset_splits)
        ]

        return num_dataset_splits, splits_indices

    def get_original_order(self, new_arr: list) -> list:
        """
        Get the original order of the data.

        Args:
            newarr (list): Array containing any kind of data that needs to be
                reset in the original order.

        Returns:
            list: new_arr in the original order.
        """
        original_order = [None] * self.total_size

        for original_index, v in zip(self.original_order, new_arr):
            original_order[original_index] = v

        if None in original_order:
            raise RuntimeError(
                f"Some elements of the original order are None, meaning that len(new_arr) ({len(new_arr)}) != len(original_array) ({self.total_size})"
            )

        return original_order

    def get_split_start_end(self, split_id: int) -> Tuple[int, int]:
        """
        Get the start and end indices of a dataset split.

        Args:
            split_id (int): The ID of the split.

        Returns:
            tuple: A tuple containing the start and end indices of the split.
        """
        self.split_start, self.split_end = self.splits[split_id]
        return self.split_start, self.split_end

    def splits_start_end_iterator(self) -> Iterator[Tuple[int, int]]:
        """
        Iterator that yields the start and end indices of each dataset split.
        Also updates the starting batch size for each split (trying to double
        the batch every time we move to a new split).

        Yields:
            tuple: A tuple containing the start and end indices of a split.
        """
        split_range = self.num_dataset_splits
        if self.total_size == 0:
            split_range = 0
        for split_id in range(split_range):
            yield self.get_split_start_end(split_id)

    def __getitem__(self, index) -> Request:
        """
        Get an item from the dataset depending on the split we are currently in.
        For instance, if we are in split 0, we will get the item at index 0, if
        we are in split 1, we will get the item at index self.split_size, etc.
        Used for dynamic batching.

        Args:
            index (int): The index of the item.

        Returns:
            Any: The item at the specified index.
        """
        return self.sorted_data[index + self.split_start]

    def __len__(self) -> int:
        """
        Get the length of current split the dataset.
        All splits have the same length, except the last one which might be
        shorter.

        Returns:
            int: The length of the dataset.
        """
        return self.split_end - self.split_start

    def __iter__(self) -> Iterator[Request]:
        """
        Iterator that yields the items of the dataset depending on the split we
        are currently in. For instance, if we are in split 0, we will get the
        items from index 0 to self.split_size, if we are in split 1, we will get
        the items from index self.split_size to 2 * self.split_size, etc. Used
        for dynamic batching.

        Yields:
            Any: The items of the dataset.
        """
        for i in range(self.split_start, self.split_end):
            yield self.sorted_data[i]

    def _sorting_criteria(self, request) -> int:
        raise NotImplementedError()


class LoglikelihoodDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request: LoglikelihoodRequest | LoglikelihoodRollingRequest) -> int:
        """
        Collates the input data for batching.

        the negative sign on len(toks) sorts descending - this has a few
        advantages:
        - time estimates will always be over not underestimates, which is
        more useful for planning
        - to know the size of a batch when going through the list, you
        know the first one is always the batch padded context length. this
        is useful to simplify the batching logic and more importantly to make
        automatic adaptive batches much much easier to implement
        - any OOMs will happen right away rather than near the end

        Args:
            x (tuple): A tuple containing the input data.

        Returns:
            tuple: A tuple containing the sorted input data.
        """
        if hasattr(request, "tokenized_continuation"):
            # We take the full context + continuation
            toks = request.tokenized_context + request.tokenized_continuation
        elif hasattr(request, "chosen_input_ids") and hasattr(request, "rejected_input_ids"):
            # We take the full context + chosen + reject
            toks = request.tokenized_context + request.chosen_input_ids + request.rejected_input_ids
        else:
            raise ValueError(
                "You passed a request for which tokenization had not happened yet. Please check your code."
            )
        return -len(toks)


class LoglikelihoodSingleTokenDataset(DynamicBatchDataset):
    def _sorting_criteria(self, request: LoglikelihoodSingleTokenRequest) -> int:
        """
        Collates the input data for batching.

        the negative sign on len(toks) sorts descending - this has a few # advantages:
        - time estimates will always be over not underestimates, which is
        more useful for planning
        - to know the size of a batch when going through the list, you
        know the first one is always the batch padded context length. this
        is useful to simplify the batching logic and more importantly to make
        automatic adaptive batches much much easier to implement
        - any OOMs will happen right away rather than near the end
        """
        # We take only the prompt, no need for the continuation (since it's a list of single tokens)
        toks = request.tokenized_context
        return -len(toks)


class GenerativeTaskDataset(DynamicBatchDataset):
    def init_split_limits(self, num_dataset_splits):
        """Initialises the split limits based on generation parameters.
        The splits are used to estimate time remaining when evaluating, and in the case of generative evaluations, to group similar samples together.

        For generative tasks, self._sorting_criteria outputs:
        - a boolean (whether the generation task uses logits)
        - a list (the stop sequences)
        - the item length (the actual size sorting factor).

        In the current function, we create evaluation groups by generation parameters (logits and eos), so that samples with similar properties get batched together afterwards.
        The samples will then be further organised by length in each split.

        Args:
            num_dataset_splits (_type_): _description_

        Returns:
            _type_: _description_
        """
        if num_dataset_splits is not None:
            logger.warning(
                "You cannot select the number of dataset splits for a generative evaluation at the moment. Automatically inferring."
            )

        if len(self.sorted_data) > 0:
            all_sorting_criterion = [self._sorting_criteria(self.sorted_data[0])[:-1]]
        splits_indices = [[0, None]]
        for ix, req in enumerate(self.sorted_data):
            current_sorting_criteria = self._sorting_criteria(req)
            current_key = current_sorting_criteria[:-1]
            if current_key not in all_sorting_criterion:
                all_sorting_criterion.append(current_key)
                splits_indices[-1][1] = ix
                splits_indices.append([ix, None])

        # We add the last split
        splits_indices[-1][1] = self.total_size

        num_dataset_splits = len(splits_indices)
        splits_indices = [tuple(e) for e in splits_indices]
        return num_dataset_splits, splits_indices

    def _sorting_criteria(self, request: GreedyUntilRequest) -> tuple[bool, bool, list, int, int]:
        """
        Collate function for generating batches.

        Args:
            x (Any): The input data.

        Returns:
            Any: The collated data.
        """
        toks = request.tokenized_context
        gen_length = request.generation_size
        # The generative task has no limit except the model context
        if gen_length is None:
            gen_length = 0
        return (
            request.do_sample,
            request.use_logits,
            tuple(request.stop_sequence),
            gen_length,
            -(len(toks) + gen_length),
        )


class GenerativeTaskDatasetNanotron(GenerativeTaskDataset):
    def __getitem__(self, index) -> Request:
        """
        Get an item from the dataset depending on the split we are currently in.
        For instance, if we are in split 0, we will get the item at index 0, if
        we are in split 1, we will get the item at index self.split_size, etc.
        Used for dynamic batching.

        Args:
            index (int): The index of the item.

        Returns:
            Any: The item at the specified index.
        """
        return index, self.sorted_data[index + self.split_start]


class GenDistributedSampler(DistributedSampler):
    """A distributed sampler that copy the last element only when drop_last is False so we keep a small padding in the batches
    as our samples are sorted by length.
    """

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            indices += [
                indices[-1]
            ] * padding_size  # This is our only change here compared to the original DistributedSampler
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

# NOTE: for JIT-compatibility, we need to be more restrictive here and use specific types instead of Iterable.
def pad_sequence(
    sequences: Union[torch.Tensor, List[torch.Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> torch.Tensor:
    r"""Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length. :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including 0). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences`), ``T`` is the length of the longest
    sequence.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): value for padded elements. Default: 0.
        padding_side (str, optional): the side to pad the sequences on.
            Default: "right".

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # JIT doesn't support `Iterable`
        if not isinstance(sequences, Iterable):
            msg = (
                "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                f"{type(sequences)}"
            )
            raise RuntimeError(msg)

        # In JIT context this leads to,
        # RuntimeError: cannot statically infer the expected size of a list in this context
        sequences = tuple(sequences)  # type: ignore[assignment]
    else:
        # For JIT, we only support Union[Tensor, Tuple[Tensor]]
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)  # type: ignore[assignment]

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value, padding_side  # type: ignore[arg-type]
    )

def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`bool` or `None`, `optional`, defaults to `None`):
            Whether you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, reqeusts: list[DPOInferenceRequest]) -> dict[str, Any]:
        features = [vars(request) for request in reqeusts]
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                # Set padding value based on the key
                if k.endswith("_input_ids"):
                    if self.pad_token_id is None:
                        raise ValueError(
                            "Padding is enabled, but the tokenizer is not configured with a padding token."
                            " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                            " before calling the trainer."
                        )
                    padding_value = self.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                elif k.endswith("_pixel_values"):
                    padding_value = 0  # TODO: check if this is correct
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Set padding side based on the key
                if k in ["prompt_input_ids", "prompt_attention_mask"]:
                    padding_side = "left"
                else:
                    padding_side = "right"

                if k.endswith("_pixel_values"):
                    dtype = torch.float32  # will be downcasted if necessary by the Trainer
                else:
                    dtype = torch.int64

                if k.endswith(("_input_ids", "_labels")):
                    for ex in features:
                        new_list = [0 if item is None else item for item in ex[k]]
                        ex[k] = new_list

                to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.endswith("_logps") and features[0][k] is not None:
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch