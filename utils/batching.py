from torch.utils.data import Sampler
import torch, random

class TokenBatchSampler(Sampler):
    """
    Groups indices so that ∑len(sample) ≤ max_tokens.
    Works with HuggingFace Datasets that already have a
    'length' column (or add it yourself).
    """
    def __init__(self, lengths, max_tokens: int, shuffle=True):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def __iter__(self):
        idxs = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(idxs)

        batch, tokens = [], 0
        for i in idxs:
            l = self.lengths[i]
            # if adding the sample would overflow the budget
            if tokens + l > self.max_tokens and batch:
                yield batch
                batch, tokens = [], 0
            batch.append(i)
            tokens += l
        if batch:
            yield batch

    def __len__(self):
        # rough estimate – not used by the Trainer
        return int(sum(self.lengths) / self.max_tokens) + 1 