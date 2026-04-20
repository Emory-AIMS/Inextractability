"""
Shared utilities: teacher-forced forward pass to extract per-token ranks and probabilities.
"""

import math
import torch
from typing import Optional


def get_token_ranks(model, tokenizer, text: str, device: Optional[str] = None):
    """
    Single teacher-forced forward pass over `text`.

    Returns:
        ranks : list[int]   ranks[i] = rank (1-indexed) of token i+1 given tokens 0..i
        probs : list[float] probs[i] = actual softmax probability of token i+1
        input_ids : list[int] tokenized sequence
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(input_ids) < 2:
        return [], [], input_ids

    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        logits = model(input_tensor).logits[0]           # [seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)   # numerically stable

    ranks = []
    probs = []
    for pos in range(1, len(input_ids)):
        pos_logits = logits[pos - 1]
        actual_token = input_ids[pos]

        sorted_indices = torch.argsort(pos_logits, descending=True)
        rank = (sorted_indices == actual_token).nonzero(as_tuple=True)[0].item() + 1
        prob = torch.exp(log_probs[pos - 1, actual_token]).item()

        ranks.append(rank)
        probs.append(prob)

    return ranks, probs, input_ids
