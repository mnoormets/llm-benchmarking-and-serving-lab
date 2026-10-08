"""Local, fixed-token sliding-window causal language model evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def evaluate_tokens(model, token_ids, window=512, stride=256, device="cpu"):
    import torch
    if not 1 <= stride < window:
        raise ValueError("Require 1 <= stride < window")
    if len(token_ids) < 2:
        raise ValueError("At least two tokens required")
    previous_end = 0
    total_nll = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(token_ids), stride):
            end = min(begin + window, len(token_ids))
            new = end - previous_end
            inputs = torch.tensor([token_ids[begin:end]], device=device)
            labels = inputs.clone()
            if new < inputs.shape[1]:
                labels[:, :-new] = -100
            scored = int((labels[:, 1:] != -100).sum().item())
            loss = float(model(input_ids=inputs, labels=labels).loss)
            if not math.isfinite(loss):
                raise ValueError("Nonfinite model loss")
            total_nll += loss * scored
            count += scored
            previous_end = end
            if end == len(token_ids):
                break
    if count != len(token_ids) - 1:
        raise AssertionError("Token coverage mismatch")
    mean = total_nll / count
    return {"scored_tokens": count, "mean_negative_log_likelihood": mean,
            "perplexity": math.exp(mean) if mean < 700 else None,
            "window": window, "stride": stride}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default="runs/perplexity.json")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    raw = Path(args.corpus).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not 1 <= len(rows) <= 1000:
        raise ValueError("Corpus must have 1 to 1000 JSONL text records")
    if not all(isinstance(row.get("text"), str) and row["text"] for row in rows):
        raise ValueError("Each record needs a nonempty text")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, trust_remote_code=False)
    # The same joined stream and tokenizer must be used for every comparison.
    tokens = tokenizer("\n\n".join(row["text"] for row in rows), add_special_tokens=False)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
                                               trust_remote_code=False, use_safetensors=True).to(args.device)
    report = evaluate_tokens(model, tokens, args.window, args.stride, args.device)
    report.update(corpus_sha256=hashlib.sha256(raw).hexdigest(),
                  token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                  model_path=args.model_path, tokenizer_path=args.tokenizer_path,
                  device=args.device, torch_version=torch.__version__)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
