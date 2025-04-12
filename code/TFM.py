# replace DataCollatorForLanguageModeling

import torch

class DataCollatorForLanguageModeling(DataCollatorMixin):    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # Modified MLM strategies
    mask_strategy = torch.bernoulli(torch.full(labels.shape, 0.50)).bool() & masked_indices
    delete_strategy = torch.bernoulli(torch.full(labels.shape, 0.10)).bool() & masked_indices & ~mask_strategy
    infill_strategy = torch.bernoulli(torch.full(labels.shape, 0.20)).bool() & masked_indices & ~mask_strategy & ~delete_strategy
    permute_strategy = torch.bernoulli(torch.full(labels.shape, 0.10)).bool() & masked_indices & ~mask_strategy & ~delete_strategy & ~infill_strategy
    
    # Token Masking
    inputs[mask_strategy] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # Token Deletion
    inputs[delete_strategy] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    # Text Infilling
    poisson_lengths = torch.poisson(torch.full(labels.shape, 3.0))
    infill_indices = infill_strategy.nonzero(as_tuple=True)[0]
    for idx in infill_indices:
        span_length = poisson_lengths[idx].item() if poisson_lengths[idx].numel() == 1 else 0
        if span_length > 0:
            inputs[idx:idx+span_length] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # Sentence Permutation
    permute_indices = permute_strategy.nonzero(as_tuple=True)[0]
    for idx in permute_indices:
        np.random.shuffle(inputs[idx])
        
    # Keep 10% Unchanged
    unchanged_strategy = ~masked_indices

    # print("TFM strategy!")