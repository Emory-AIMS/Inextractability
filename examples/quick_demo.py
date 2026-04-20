"""
Quick demo: runs both Algorithm 2 and Algorithm 3 on a built-in sample text.

Usage:
    python examples/quick_demo.py                  # uses GPT-2 by default
    python examples/quick_demo.py --model gpt2-xl  # specify a different model
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from inextractability import estimate_extraction_cost, estimate_greedy_rate

SAMPLE_TEXTS = [
    (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump."
    ),
    (
        "We hold these truths to be self-evident, that all men are created equal, "
        "that they are endowed by their Creator with certain unalienable Rights, "
        "that among these are Life, Liberty and the pursuit of Happiness."
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Quick demo of inextractability measurement")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model (default: gpt2)")
    parser.add_argument("--l", type=int, default=5, help="Sliding window length (default: 5 for demo)")
    parser.add_argument("--m", type=int, default=20, help="Rank threshold (default: 20)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    print(f"\nDataset: {len(SAMPLE_TEXTS)} sequences")
    for i, t in enumerate(SAMPLE_TEXTS):
        print(f"  [{i}] {t[:60]}...")

    # --- Algorithm 2 ---
    result_b = estimate_extraction_cost(model, tokenizer, SAMPLE_TEXTS, l=args.l, m=args.m)
    print(f"\n{'='*50}")
    print(f"Algorithm 2: Rank-Aware Estimation of Extraction Cost")
    print(f"{'='*50}")
    print(f"  b (inextractability):  {result_b['b']:.4f} bits")
    print(f"  p* (max window prob):  {result_b['p_star']:.2e}")
    print(f"  worst sequence:        {result_b['worst_seq']}")
    print(f"  worst span (tokens):   {result_b['worst_span']}")

    # --- Algorithm 3 ---
    result_g = estimate_greedy_rate(model, tokenizer, SAMPLE_TEXTS, l=args.l)
    print(f"\n{'='*50}")
    print(f"Algorithm 3: Efficient Estimation for Greedy Generation")
    print(f"{'='*50}")
    print(f"  eta (greedy rate):     {result_g['eta']:.4f}")
    print(f"  n_extractable:         {result_g['n_extractable']}")
    print(f"  n_total windows:       {result_g['n_total']}")

    print(f"\nDone! See the paper for details on interpreting these metrics.")


if __name__ == "__main__":
    main()
