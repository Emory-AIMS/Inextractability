"""
Core algorithms from "Beyond Indistinguishability: Measuring Extraction Risk in LLM APIs".

Algorithm 2 – Rank-Aware Estimation of Extraction Cost  → estimate_extraction_cost
Algorithm 3 – Efficient Estimation for Greedy Generation → estimate_greedy_rate
"""

import math
from typing import List, Optional, Union

from .utils import get_token_ranks


# ---------------------------------------------------------------------------
# Algorithm 2 – single-sequence helper
# ---------------------------------------------------------------------------

def _extraction_cost_single(
    ranks: list,
    probs: list,
    l: int,
    m: int,
) -> dict:
    """Compute extraction cost for a single pre-ranked sequence."""
    n = len(ranks)
    if n < l:
        return None  # sequence too short, skip

    max_log2_p = -math.inf
    worst_span = None

    for i in range(n - l + 1):
        log2_p = 0.0
        for j in range(i, i + l):
            r = ranks[j]
            if r <= m:
                log2_p -= math.log2(r)          # log2(1/r)
            else:
                p = probs[j]
                log2_p += math.log2(p) if p > 0.0 else -math.inf

        if log2_p > max_log2_p:
            max_log2_p = log2_p
            worst_span = (i + 1, i + l)         # 1-indexed, inclusive

    p_star = 2.0 ** max_log2_p if max_log2_p > -math.inf else 0.0
    b = -max_log2_p
    return {"b": b, "p_star": p_star, "worst_span": worst_span}


# ---------------------------------------------------------------------------
# Algorithm 2 – public API (matches paper: iterates over D_pro)
# ---------------------------------------------------------------------------

def estimate_extraction_cost(
    model,
    tokenizer,
    texts: Union[str, List[str]],
    l: int = 50,
    m: int = 20,
    device: Optional[str] = None,
) -> dict:
    """
    Algorithm 2: Rank-Aware Estimation of Extraction Cost.

    Iterates over every sequence in the protected dataset D_pro, and for
    every l-gram window z computes
        p_z = prod_t  (1/r_t  if r_t <= m  else  P_t(z_t)).
    Returns  b = -log2(max_z p_z)  across ALL sequences and windows.

    Args:
        model:     loaded HuggingFace CausalLM
        tokenizer: corresponding HuggingFace tokenizer
        texts:     a single string or a list of strings (D_pro)
        l:         sliding-window length (default 50)
        m:         rank threshold; tokens with rank > m use their actual probability
        device:    torch device string (auto-detected from model if None)

    Returns:
        {
          "b":          float  – inextractability level in bits (higher = harder to extract),
          "p_star":     float  – max extraction probability across all sequences and windows,
          "worst_seq":  int    – 0-indexed sequence in D_pro containing the easiest window,
          "worst_span": tuple  – 1-indexed (start, end) token positions within that sequence,
          "per_sequence": list – per-sequence results [{"b", "p_star", "worst_span"}, ...],
        }

    Raises:
        ValueError if no sequence has at least l tokens.
    """
    if isinstance(texts, str):
        texts = [texts]

    global_max_log2_p = -math.inf
    global_worst_span = None
    global_worst_seq = None
    per_sequence = []

    for seq_idx, text in enumerate(texts):
        ranks, probs, _ = get_token_ranks(model, tokenizer, text, device)

        result = _extraction_cost_single(ranks, probs, l, m)
        if result is None:
            per_sequence.append(None)  # sequence too short
            continue

        per_sequence.append(result)

        if result["p_star"] > 0.0:
            log2_p = math.log2(result["p_star"])
        else:
            log2_p = -math.inf

        if log2_p > global_max_log2_p:
            global_max_log2_p = log2_p
            global_worst_span = result["worst_span"]
            global_worst_seq = seq_idx

    if global_worst_seq is None:
        raise ValueError(
            f"No sequence in D_pro has at least {l} tokens."
        )

    p_star = 2.0 ** global_max_log2_p if global_max_log2_p > -math.inf else 0.0
    b = -global_max_log2_p

    return {
        "b": b,
        "p_star": p_star,
        "worst_seq": global_worst_seq,
        "worst_span": global_worst_span,
        "per_sequence": per_sequence,
    }


# ---------------------------------------------------------------------------
# Algorithm 3 – single-sequence helper
# ---------------------------------------------------------------------------

def _greedy_rate_single(ranks: list, l: int) -> dict:
    """Compute greedy extractable rate for a single pre-ranked sequence."""
    n = len(ranks)
    if n < l:
        return None  # sequence too short, skip

    n_total = n - l + 1
    n_extractable = 0
    i = 0

    while i <= n - l:
        fail_pos = None
        for j in range(i, i + l):
            if ranks[j] != 1:
                fail_pos = j
                break

        if fail_pos is None:
            n_extractable += 1
            i += 1
        else:
            i = fail_pos + 1

    eta = n_extractable / n_total if n_total > 0 else 0.0
    return {"eta": eta, "n_extractable": n_extractable, "n_total": n_total}


# ---------------------------------------------------------------------------
# Algorithm 3 – public API (matches paper: iterates over D_pro)
# ---------------------------------------------------------------------------

def estimate_greedy_rate(
    model,
    tokenizer,
    texts: Union[str, List[str]],
    l: int = 50,
    device: Optional[str] = None,
) -> dict:
    """
    Algorithm 3: Efficient Estimation for Greedy Generation.

    Iterates over every sequence in the protected dataset D_pro.  For each
    sequence, counts l-gram windows where every token has rank 1
    (i.e. greedy-decodable), using a skip optimisation.  Returns the
    overall extractable rate across ALL sequences.

    Args:
        model:     loaded HuggingFace CausalLM
        tokenizer: corresponding HuggingFace tokenizer
        texts:     a single string or a list of strings (D_pro)
        l:         sliding-window length (default 50)
        device:    torch device string (auto-detected from model if None)

    Returns:
        {
          "eta":           float – greedy extractable rate across D_pro,
          "n_extractable": int   – total greedy-extractable windows,
          "n_total":       int   – total windows across all sequences,
          "per_sequence":  list  – per-sequence results [{"eta", "n_extractable", "n_total"}, ...],
        }

    Raises:
        ValueError if no sequence has at least l tokens.
    """
    if isinstance(texts, str):
        texts = [texts]

    total_extractable = 0
    total_windows = 0
    per_sequence = []

    for text in texts:
        ranks, _, _ = get_token_ranks(model, tokenizer, text, device)

        result = _greedy_rate_single(ranks, l)
        if result is None:
            per_sequence.append(None)
            continue

        per_sequence.append(result)
        total_extractable += result["n_extractable"]
        total_windows += result["n_total"]

    if total_windows == 0:
        raise ValueError(
            f"No sequence in D_pro has at least {l} tokens."
        )

    eta = total_extractable / total_windows

    return {
        "eta": eta,
        "n_extractable": total_extractable,
        "n_total": total_windows,
        "per_sequence": per_sequence,
    }
