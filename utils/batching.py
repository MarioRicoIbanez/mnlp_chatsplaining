from torch.utils.data import Sampler
import torch
import random
from collections import deque
import math

class SmartPaddingTokenBatchSampler(Sampler):
    """
    Groups indices to mitigate OOMs due to padding.
    - Samples longer than `long_sample_threshold` are processed individually.
    - For shorter samples, batches are formed such that:
        N * L_max <= max_tokens_effective_budget
      where N is the batch size and L_max is the length of the longest sample in that batch.
    - Batch sizes (N) are constrained to be powers of two (or can be just even).
    """
    def __init__(self,
                 lengths,
                 max_tokens_effective_budget: int, # e.g., 8000, represents N * L_max
                 long_sample_threshold: int = 4000, # Samples > this are batched individually
                 min_batch_size: int = 1,         # Min samples in a batch
                 constrain_to_power_of_two: bool = True, # True: BS is 1,2,4,8... False: BS is even (or >= min_bs)
                 shuffle=False,
                 enable_sanity_checks=True,
                 max_sequence_len_overall: int = None # Absolute max length any sample can have after tokenization
                ):

        if not isinstance(lengths, list) and not isinstance(lengths, torch.Tensor):
            try:
                lengths = list(lengths)
            except TypeError:
                raise ValueError("lengths must be a list, torch.Tensor, or an iterable convertible to a list.")

        self.lengths = lengths
        self.max_tokens_effective_budget = max_tokens_effective_budget
        self.long_sample_threshold = long_sample_threshold
        self.min_batch_size = min_batch_size
        self.constrain_to_power_of_two = constrain_to_power_of_two
        self.shuffle = shuffle
        self.enable_sanity_checks = enable_sanity_checks
        self.max_sequence_len_overall = max_sequence_len_overall if max_sequence_len_overall is not None else max_tokens_effective_budget


        self.long_indices_group = []
        self.short_indices_group = []
        self.skipped_indices_too_long = [] # Longer than even the model/budget can handle

        for i in range(len(self.lengths)):
            l = self.lengths[i]
            if l > self.max_sequence_len_overall:
                self.skipped_indices_too_long.append(i)
                print(f"Warning: Sample {i} with length {l} exceeds max_sequence_len_overall "
                      f"{self.max_sequence_len_overall} and will be skipped entirely.")
                continue

            if l > self.long_sample_threshold:
                if l > self.max_tokens_effective_budget : # Check if it can even fit as a batch of 1
                    # This sample is longer than threshold AND too long for budget as BS=1
                     print(f"Warning: Long Sample {i} (length {l}) also exceeds max_tokens_effective_budget "
                           f"{self.max_tokens_effective_budget} when considered as batch_size=1. Will be skipped.")
                     self.skipped_indices_too_long.append(i)
                else:
                    self.long_indices_group.append(i)
            else:
                self.short_indices_group.append(i)
        
        if self.skipped_indices_too_long:
            print(f"Total skipped indices due to excessive length: {len(self.skipped_indices_too_long)}")


    def _get_target_batch_size(self, max_allowed_bs: int) -> int:
        """
        Given max_allowed_bs (due to L_max * N <= budget),
        find the largest valid batch size (power of 2 or even)
        that is <= max_allowed_bs and >= self.min_batch_size.
        Returns 0 if no such size found.
        """
        target_bs = 0
        if self.constrain_to_power_of_two:
            # Find largest power of 2 <= max_allowed_bs
            if max_allowed_bs >= 1:
                # floor(log2(max_allowed_bs)) gives the exponent. 2^exponent.
                target_bs = 1 << (max_allowed_bs.bit_length() - 1)
                # Alternative for older Python or clarity:
                # p = 0
                # while (1 << p) <= max_allowed_bs:
                #    target_bs = 1 << p
                #    p += 1
            else:
                target_bs = 0 # Cannot even fit one sample if max_allowed_bs is 0
        else: # Just even (or any if min_batch_size allows)
            target_bs = max_allowed_bs
            if target_bs % 2 != 0 and target_bs > 1 : # ensure evenness unless it's 1
                 target_bs -=1

        # Ensure it meets min_batch_size, unless target_bs became 0
        if target_bs < self.min_batch_size and target_bs !=0 : # if target_bs became 0, it means it can't fit
            return 0 # Cannot satisfy min_batch_size with this constraint
        
        # If min_batch_size forced target_bs to 0 previously, but max_allowed_bs was >= min_batch_size
        # Example: min_bs=3, power_of_2=true, max_allowed_bs=3. Power_of_2 would give 2. This fails min_bs.
        # This part is tricky. The logic above for power_of_2 should find *a* power of 2.
        # Then we check if that power_of_2 meets min_batch_size.
        if self.constrain_to_power_of_two:
             # Re-check: find largest power of 2 that is also >= min_batch_size
             current_best_pow2 = 0
             p = 0
             while True:
                 pow2_val = 1 << p
                 if pow2_val > max_allowed_bs:
                     break
                 if pow2_val >= self.min_batch_size:
                     current_best_pow2 = pow2_val
                 elif pow2_val == 1 and self.min_batch_size == 1 and max_allowed_bs >=1: # Special case for min_batch_size = 1
                     current_best_pow2 = 1

                 if pow2_val >= max_allowed_bs : # Optimization: no need to check higher powers if already > max_allowed_bs
                    if (1 << (p+1)) > max_allowed_bs and current_best_pow2 == 0 and pow2_val <= max_allowed_bs and pow2_val >= self.min_batch_size:
                         # handles case where largest pow2 < max_allowed_bs is also < min_batch_size,
                         # but max_allowed_bs itself could accommodate a min_batch_size which is a power of 2
                         pass # current_best_pow2 should be set if one exists
                    # break # (This break might be too early if current_best_pow2 isn't the true largest)

                 # Safety break for p to avoid infinite loop if logic is flawed elsewhere
                 if p > 10: # Max batch size 1024, quite high
                     break
                 p += 1
             target_bs = current_best_pow2
        else: # Not power of two constraint
            if max_allowed_bs < self.min_batch_size:
                return 0
            target_bs = max_allowed_bs
            if target_bs % 2 != 0 and self.min_batch_size > 1: # If min_batch_size is 1, odd is fine
                 target_bs -=1
            if target_bs < self.min_batch_size: # After making it even, it might fall below min_batch_size
                return 0
        
        return target_bs


    def __iter__(self):
        yielded_indices_this_epoch = set()

        # 1. Process long samples individually
        long_indices_iter = list(self.long_indices_group)
        if self.shuffle:
            random.shuffle(long_indices_iter)

        for long_idx in long_indices_iter:
            if self.enable_sanity_checks and long_idx in yielded_indices_this_epoch:
                raise RuntimeError(f"Sanity Check FAILED (long): Index {long_idx} already yielded.")
            yield [long_idx]
            if self.enable_sanity_checks:
                yielded_indices_this_epoch.add(long_idx)

        # 2. Process short samples
        # Sort by length (descending) to pick the longest first for a new batch
        # This ensures L_max for the batch is set by the first sample.
        short_indices_sorted = sorted(self.short_indices_group, key=lambda i: self.lengths[i], reverse=True)
        
        # If shuffling, we shuffle this presorted list. This maintains groups of similarly-lengthed items
        # somewhat together, which can be beneficial for this strategy, but still provides epoch-level shuffle.
        # Or, shuffle *before* sorting if we want pure random first item (less optimal for this strategy).
        # For this strategy, it's better to shuffle *after* sorting, or just use the sorted order.
        # Let's go with shuffling the sorted list for now.
        if self.shuffle:
            # A full shuffle of short_indices_sorted might break the "longest first" heuristic for batch start.
            # Better: shuffle self.short_indices_group then sort. Or operate on the sorted list.
            # Let's use the sorted list and pop from it. If shuffle, it's applied to the initial pool.
            # The key is: once we start a batch, L_max is fixed by the first item from sorted list.
            
            # Create the pool from pre-shuffled short_indices_group, then sort for processing.
            # No, the current approach: sort all short samples, then make a pool.
            # If shuffle is True, the order within `pool` will be random *after* taking elements.
            # This is complex. Let's simplify:
            # Always sort short_indices_group by length descending to form the initial pool.
            # The `shuffle` flag will apply to the order these batches are yielded eventually if we collect all
            # batches and then shuffle them. Or, it applies to how `long_indices_iter` and `short_indices_group`
            # are initially formed.
            # For now, `shuffle` affects order of long_indices and initial order of short_indices if not sorted.
            # Given the logic, sorting short_indices is crucial for `L_max` determination.
            pass # short_indices_sorted is already prepared

        pool = deque(short_indices_sorted)
        carry_over = deque() # For samples that couldn't form a batch in a previous attempt

        while pool or carry_over:
            current_batch_indices = []
            
            # Try to start a batch: prioritize carry_over, then pool
            # The crucial part is that the *first* sample taken for a new batch
            # must come from the sorted pool (or carry_over derived from it)
            # to correctly set L_max.

            items_to_consider_for_batch_start = deque()
            if carry_over:
                # Sort carry_over by length descending to pick its longest first
                items_to_consider_for_batch_start.extend(
                    sorted(list(carry_over), key=lambda i: self.lengths[i], reverse=True)
                )
                carry_over.clear() # Will be repopulated if not used
            # Then add from pool (which is already sorted)
            # We only need one candidate to start, the rest can be unsorted from pool/carry_over
            # as long as their length <= L_max
            
            if not items_to_consider_for_batch_start and pool:
                # If carry_over was empty, the pool is our source for starting a batch
                # (pool is already sorted by length desc)
                if pool[0] not in yielded_indices_this_epoch: # Check before popping
                    items_to_consider_for_batch_start.append(pool.popleft())
                else: # already yielded, should not happen if logic is correct
                    _ = pool.popleft() # discard
                    continue

            if not items_to_consider_for_batch_start:
                break # No more items to start a batch

            first_idx = items_to_consider_for_batch_start.popleft()
            # Put rest of items_to_consider_for_batch_start back into carry_over
            # so they are tried next if this batch fails, or added to current batch
            carry_over.extendleft(reversed(items_to_consider_for_batch_start))


            if self.enable_sanity_checks and first_idx in yielded_indices_this_epoch:
                # This sample was yielded, perhaps as a long sample or in a previous short batch.
                # Or it was in carry_over and got re-yielded.
                # This indicates a potential logic flaw in how yielded_indices_this_epoch or carry_over is managed.
                print(f"Warning/Debug: Sample {first_idx} (from short/carry_over) already in yielded_indices_this_epoch. Skipping.")
                continue

            L_max_for_this_batch = self.lengths[first_idx]
            
            # Max number of samples of length L_max_for_this_batch that can fit
            if L_max_for_this_batch == 0: # Avoid division by zero for zero-length (empty) samples
                max_allowed_bs_for_L_max = float('inf') # Effectively, many can fit
            else:
                max_allowed_bs_for_L_max = self.max_tokens_effective_budget // L_max_for_this_batch
            
            if max_allowed_bs_for_L_max == 0 : # This first_idx itself is too long for the budget
                print(f"Debug: first_idx {first_idx} (len {L_max_for_this_batch}) too long for budget "
                      f"{self.max_tokens_effective_budget}. Effective max_bs=0. Adding to carry_over.")
                if first_idx not in yielded_indices_this_epoch: # only carry over if not already processed
                    carry_over.append(first_idx) # Try later, maybe alone if min_bs allows
                continue


            target_bs = self._get_target_batch_size(max_allowed_bs_for_L_max)

            if target_bs == 0: # Cannot form a valid batch starting with this sample
                if first_idx not in yielded_indices_this_epoch:
                     carry_over.append(first_idx) # Defer it
                continue

            current_batch_indices.append(first_idx)
            
            # Now, fill up the rest of the batch up to target_bs
            # from carry_over (items whose len <= L_max_for_this_batch)
            # and then from pool (items whose len <= L_max_for_this_batch)
            
            temp_requeue_carry = deque()
            while len(current_batch_indices) < target_bs and carry_over:
                candidate_idx = carry_over.popleft()
                if self.enable_sanity_checks and candidate_idx in yielded_indices_this_epoch:
                    continue
                if self.lengths[candidate_idx] <= L_max_for_this_batch: # Important check!
                    current_batch_indices.append(candidate_idx)
                else: # Too long for *this* batch's L_max, requeue
                    temp_requeue_carry.append(candidate_idx)
            carry_over.extendleft(reversed(temp_requeue_carry)) # Put back unfit items

            temp_requeue_pool = deque()
            while len(current_batch_indices) < target_bs and pool:
                candidate_idx = pool.popleft()
                if self.enable_sanity_checks and candidate_idx in yielded_indices_this_epoch:
                    continue
                # Since pool was sorted by length descending, and first_idx came from its head (or carry_over's head),
                # subsequent items from pool *should* be <= L_max_for_this_batch.
                # However, if shuffle happened or carry_over items were mixed, this check is good.
                if self.lengths[candidate_idx] <= L_max_for_this_batch:
                     current_batch_indices.append(candidate_idx)
                else: # This sample is too long for current batch's L_max constraint
                      # This should ideally not happen if pool is strictly sorted and L_max is from its head.
                      # But if pool gets shuffled or carry_over items are longer, this path is taken.
                    temp_requeue_pool.append(candidate_idx)
            pool.extendleft(reversed(temp_requeue_pool)) # Put back unfit items

            # Final check on batch
            if len(current_batch_indices) == target_bs:
                if self.enable_sanity_checks:
                    for idx_cb in current_batch_indices:
                        if idx_cb in yielded_indices_this_epoch:
                            raise RuntimeError(f"Sanity Check FAILED (short batch formation): Index {idx_cb} already yielded.")
                    yielded_indices_this_epoch.update(current_batch_indices)
                yield current_batch_indices
            elif len(current_batch_indices) >= self.min_batch_size:
                # We couldn't reach target_bs, but have enough for min_batch_size.
                # Try to make a smaller valid batch (e.g. if target was 8, but we only got 5, try for 4 or 2)
                actual_len = len(current_batch_indices)
                # Max allowed for this *actual_len* is actual_len itself. We want largest valid size <= actual_len.
                final_bs_for_partial = self._get_target_batch_size(actual_len)

                if final_bs_for_partial > 0 and final_bs_for_partial <= actual_len :
                    batch_to_yield = current_batch_indices[:final_bs_for_partial]
                    overflow_samples = current_batch_indices[final_bs_for_partial:]
                    
                    if self.enable_sanity_checks:
                        for idx_cb in batch_to_yield:
                            if idx_cb in yielded_indices_this_epoch:
                                raise RuntimeError(f"Sanity Check FAILED (partial short batch): Index {idx_cb} already yielded.")
                        yielded_indices_this_epoch.update(batch_to_yield)
                    yield batch_to_yield
                    
                    for item in reversed(overflow_samples): # Add to front of carry_over
                         if item not in yielded_indices_this_epoch:
                            carry_over.appendleft(item)
                else: # Cannot form smaller valid batch, put all back
                    for item in reversed(current_batch_indices):
                        if item not in yielded_indices_this_epoch:
                            carry_over.appendleft(item)
            else: # Less than min_batch_size, put all back
                for item in reversed(current_batch_indices):
                    if item not in yielded_indices_this_epoch:
                        carry_over.appendleft(item)
        
        # Process any remaining in carry_over that couldn't form batches
        # These might be single items that couldn't meet min_batch_size with others
        # or couldn't find partners.
        final_carry_list = sorted(list(carry_over), key=lambda i: self.lengths[i], reverse=True)
        temp_final_batch = []
        for final_idx in final_carry_list:
            if final_idx in yielded_indices_this_epoch:
                continue
            
            # Try to yield as batch of 1 if min_batch_size is 1
            if self.min_batch_size == 1:
                # Check budget for batch size 1 for this sample
                if self.lengths[final_idx] <= self.max_tokens_effective_budget:
                    if self.enable_sanity_checks:
                        if final_idx in yielded_indices_this_epoch:
                            raise RuntimeError(f"Sanity Check FAILED (final carry): Index {final_idx} already yielded.")
                        yielded_indices_this_epoch.add(final_idx)
                    yield [final_idx]
                else:
                    print(f"Info: Final carry_over sample {final_idx} (len {self.lengths[final_idx]}) "
                          f"could not be yielded as BS=1 due to budget {self.max_tokens_effective_budget}.")
            else: # min_batch_size > 1, these can't be yielded alone
                 print(f"Info: Final carry_over sample {final_idx} (len {self.lengths[final_idx]}) "
                       f"not yielded as min_batch_size is {self.min_batch_size}.")


        if self.enable_sanity_checks:
            total_initial_processable = len(self.long_indices_group) + len(self.short_indices_group)
            if len(yielded_indices_this_epoch) != total_initial_processable:
                print(f"Sanity Check INFO: Not all processable indices were yielded. "
                      f"Initial: {total_initial_processable}, Yielded: {len(yielded_indices_this_epoch)}. "
                      f"Un-yielded count: {total_initial_processable - len(yielded_indices_this_epoch)}")


    def __len__(self):
        # Rough estimate, can be complex to calculate accurately
        # Count long samples as individual batches
        num_batches = len(self.long_indices_group)
        
        # Estimate for short samples
        if self.short_indices_group:
            avg_len_short = sum(self.lengths[i] for i in self.short_indices_group) / len(self.short_indices_group)
            if avg_len_short > 0:
                # Estimate BS based on avg_len_short
                # max_bs_for_avg = self.max_tokens_effective_budget // avg_len_short
                # estimated_bs = self._get_target_batch_size(int(max_bs_for_avg if max_bs_for_avg > 0 else 1))
                # if estimated_bs > 0:
                #    num_batches += math.ceil(len(self.short_indices_group) / estimated_bs)
                # else: # If estimated_bs is 0, assume they are batched individually if possible
                #    num_batches += len(self.short_indices_group) # Fallback: worst case
                
                # Simpler estimate: assume an average batch size of around min_batch_size or slightly more
                # This is very rough.
                effective_min_bs = max(self.min_batch_size, 2 if self.constrain_to_power_of_two and self.min_batch_size <=1 else 1)

                num_batches += math.ceil(len(self.short_indices_group) / effective_min_bs)

            else: # All short samples have length 0
                num_batches += 1 # Assume they all fit in one large batch if many
        return num_batches if num_batches > 0 else 1