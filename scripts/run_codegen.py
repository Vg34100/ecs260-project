import argparse
import json
import time
from pathlib import Path

from humaneval import load_humaneval


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def dummy_complete(prompt: str) -> str:
    # Minimal placeholder completion
    return "\n    pass\n"


def generate_with_transformers(prompt: str, model_path: str, max_new_tokens: int) -> str:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise RuntimeError("transformers/torch not available") from e

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text[len(prompt) :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code generation on HumanEval")
    parser.add_argument("--dataset", default="data/HumanEval.jsonl")
    parser.add_argument("--prompt", default="prompts/codegen_base.txt")
    parser.add_argument("--model", default="dummy", choices=["dummy", "transformers"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default="runs/codegen.jsonl")
    args = parser.parse_args()

    prompt_prefix = load_prompt(args.prompt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for item in load_humaneval(args.dataset, limit=args.limit):
            task_id = item.get("task_id")
            base_prompt = item.get("prompt", "")
            full_prompt = prompt_prefix + base_prompt

            for r in range(args.repeats):
                if args.model == "dummy":
                    completion = dummy_complete(full_prompt)
                else:
                    if not args.model_path:
                        raise SystemExit("--model-path is required for transformers")
                    completion = generate_with_transformers(
                        full_prompt, args.model_path, args.max_new_tokens
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


if __name__ == "__main__":
    main()
