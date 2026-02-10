import argparse
import json
import os
import time
from pathlib import Path

from humaneval import load_humaneval


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def dummy_complete(prompt: str) -> str:
    # Minimal placeholder completion
    return "\n    pass\n"


def load_transformers(model_path: str, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise RuntimeError("transformers/torch not available") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_with_transformers(
    tokenizer, model, prompt: str, max_new_tokens: int, device: str, stop_strings
) -> str:
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
    completion = full_text[len(prompt) :]
    for s in stop_strings:
        idx = completion.find(s)
        if idx != -1:
            completion = completion[:idx]
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code generation on HumanEval")
    parser.add_argument("--dataset", default="data/HumanEval.jsonl")
    parser.add_argument("--prompt", default="prompts/codegen_base.txt")
    parser.add_argument("--model", default="dummy", choices=["dummy", "transformers"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--offline", action="store_true", help="use cached models only")
    parser.add_argument("--out", default="runs/codegen.jsonl")
    args = parser.parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    prompt_prefix = load_prompt(args.prompt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        tokenizer, model = load_transformers(args.model_path, device)
        print(f"Using device: {device}")

    stop_strings = ["\n\ndef ", "\n\nclass ", "\n\n\n"]

    with out_path.open("w", encoding="utf-8") as out:
        for idx, item in enumerate(load_humaneval(args.dataset, limit=args.limit), start=1):
            task_id = item.get("task_id")
            base_prompt = item.get("prompt", "")
            full_prompt = prompt_prefix + base_prompt

            for r in range(args.repeats):
                print(f"Task {idx}/{args.limit} {task_id} repeat {r+1}/{args.repeats}...")
                if args.model == "dummy":
                    completion = dummy_complete(full_prompt)
                else:
                    completion = generate_with_transformers(
                        tokenizer, model, full_prompt, args.max_new_tokens, device, stop_strings
                    )

                record = {
                    "task_id": task_id,
                    "prompt": full_prompt,
                    "completion": completion,
                    "model": args.model,
                    "model_path": args.model_path,
                    "repeat": r,
                    "timestamp": time.time(),
                }
                out.write(json.dumps(record) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
