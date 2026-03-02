"""Run test generation on MBPP-style JSONL."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib import request


def load_records(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_transformers(model_path: str, device: str, offline: bool):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise RuntimeError("transformers/torch not available") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=offline)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=offline
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=offline)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_with_transformers(tokenizer, model, prompt: str, max_new_tokens: int, device: str) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text[len(prompt) :].strip()


def generate_with_ollama(model: str, prompt: str, url: str, max_new_tokens: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url.rstrip("/") + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode("utf-8")
    result = json.loads(body)
    return result.get("response", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test generation")
    parser.add_argument("--dataset", default="datasets/nominal/mbpp_wtest.jsonl")
    parser.add_argument("--model", default="dummy", choices=["dummy", "transformers", "ollama"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--ollama-model", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--offline", action="store_true", help="use cached models only")
    parser.add_argument("--out", default="runs/mbpp_testgen.jsonl")
    args = parser.parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer = None
    model = None
    device = args.device
    if args.model == "transformers":
        if not args.model_path:
            raise SystemExit("--model-path is required for transformers")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        tokenizer, model = load_transformers(args.model_path, device, args.offline)
    elif args.model == "ollama":
        if not args.ollama_model:
            raise SystemExit("--ollama-model is required for ollama")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for idx, item in enumerate(load_records(args.dataset), start=1):
            if args.limit and idx > args.limit:
                break
            task_id = item.get("task_id")
            prompt = item.get("prompt", "")

            test_prompt = (
                "You are given a Python function signature and docstring. "
                "Write 3-5 pytest-style test cases using only assert statements. "
                "Return only the test code.\n\n"
                f"{prompt}\n\nTests:"
            )

            for r in range(args.repeats):
                print(f"Task {idx}/{args.limit} {task_id} repeat {r+1}/{args.repeats}")
                if args.model == "dummy":
                    completion = ""
                elif args.model == "transformers":
                    completion = generate_with_transformers(
                        tokenizer, model, test_prompt, args.max_new_tokens, device
                    )
                else:
                    completion = generate_with_ollama(
                        args.ollama_model,
                        test_prompt,
                        args.ollama_url,
                        args.max_new_tokens,
                    )

                record = {
                    "task_id": task_id,
                    "prompt": test_prompt,
                    "completion": completion,
                    "model": args.model,
                    "model_path": args.model_path,
                    "repeat": r,
                    "timestamp": time.time(),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
