"""
Example: Algorithm 2 – Rank-Aware Estimation of Extraction Cost.

Usage:
    python examples/estimate_b.py --model gpt2 --text "some text here" --l 50 --m 20
    python examples/estimate_b.py --model gpt2 --file data.txt --l 50 --m 20
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from inextractability import estimate_extraction_cost


def parse_args():
    p = argparse.ArgumentParser(description="Estimate inextractability level b (Algorithm 2)")
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text to evaluate")
    group.add_argument("--file", help="Text file to evaluate (one sequence per line)")
    p.add_argument("--l", type=int, default=50, help="Sliding window length (default: 50)")
    p.add_argument("--m", type=int, default=20, help="Rank threshold (default: 20)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.text:
        texts = [args.text]
    else:
        with open(args.file, "r") as f:
            texts = [line.strip() for line in f if line.strip()]

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    print(f"Evaluating {len(texts)} sequence(s)...")
    result = estimate_extraction_cost(model, tokenizer, texts, l=args.l, m=args.m)

    print(f"\n=== Algorithm 2: Extraction Cost (over D_pro) ===")
    print(f"  b (inextractability):  {result['b']:.4f} bits")
    print(f"  p* (max window prob):  {result['p_star']:.2e}")
    print(f"  worst sequence index:  {result['worst_seq']}")
    print(f"  worst span (tokens):   {result['worst_span']}")

    if len(texts) > 1:
        print(f"\n  --- Per-sequence breakdown ---")
        for i, r in enumerate(result["per_sequence"]):
            if r is None:
                print(f"  seq {i}: skipped (too short)")
            else:
                print(f"  seq {i}: b = {r['b']:.4f} bits")


if __name__ == "__main__":
    main()
