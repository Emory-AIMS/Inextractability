"""
Example: Algorithm 3 – Efficient Estimation for Greedy Generation.

Usage:
    python examples/estimate_greedy_rate.py --model gpt2 --text "some text here" --l 50
    python examples/estimate_greedy_rate.py --model gpt2 --file data.txt --l 50
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from inextractability import estimate_greedy_rate


def parse_args():
    p = argparse.ArgumentParser(description="Estimate greedy extractable rate η (Algorithm 3)")
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single text to evaluate")
    group.add_argument("--file", help="Text file to evaluate (one sequence per line)")
    p.add_argument("--l", type=int, default=50, help="Sliding window length (default: 50)")
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
    result = estimate_greedy_rate(model, tokenizer, texts, l=args.l)

    print(f"\n=== Algorithm 3: Greedy Extractable Rate (over D_pro) ===")
    print(f"  η  (greedy rate):      {result['eta']:.4f}")
    print(f"  n_extractable:         {result['n_extractable']}")
    print(f"  n_total windows:       {result['n_total']}")

    if len(texts) > 1:
        print(f"\n  --- Per-sequence breakdown ---")
        for i, r in enumerate(result["per_sequence"]):
            if r is None:
                print(f"  seq {i}: skipped (too short)")
            else:
                print(f"  seq {i}: η = {r['eta']:.4f}  ({r['n_extractable']}/{r['n_total']})")


if __name__ == "__main__":
    main()
