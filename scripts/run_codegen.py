import argparse
import json
import os
import re
import time
from pathlib import Path

from humaneval import load_humaneval
from validaity import check_python_syntax

def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def dummy_complete(prompt: str) -> str:
    # Minimal placeholder completion
    return "\n    pass\n"


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

def extract_code_for_parse(text: str) -> str:
    """
    Extracts code from a completion. If fenced code exists, use the first fenced block.
    Otherwise, use raw text.
    """
    if "```" not in text:
        return text.strip()

    parts = text.split("```")
    # parts: [before, fence_lang_and_code, after, ...]
    # We take the first fenced block content.
    if len(parts) >= 3:
        fenced = parts[1]
        # If it starts with 'python\n', drop the first line.
        lines = fenced.splitlines()
        if lines and lines[0].strip().lower().startswith("python"):
            lines = lines[1:]
        return "\n".join(lines).strip()

    return text.strip()


def cleanup_completion(completion: str) -> str:
    # Remove FIM placeholder tokens that can leak into final text.
    completion = re.sub(r"<\|fim_[^|]*\|>", "", completion)
    completion = completion.replace("<|cursor|>", "")

    # If the model wrapped code in markdown fences, keep only fenced body.
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", completion, flags=re.DOTALL)
    if fence_match:
        completion = fence_match.group(1)
    completion = completion.replace("```python", "").replace("```", "")

    # Not entirely necessary, but if there are common trailing patterns that indicate the model is starting to write test cases or explanations, we can cut those off to keep just the function body.
    # Drop common trailing junk patterns beyond the target function body.
    trailing_markers = [
        "\n# Test cases",
        "\n# Tests",
        "\nassert ",
        "\nprint(",
        "\nif __name__ ==",
        "\nimport unittest",
        "\nclass Test",
        "\n**Explanation",
        "\nExplanation:",
    ]
    for marker in trailing_markers:
        idx = completion.find(marker)
        if idx != -1:
            completion = completion[:idx]

    return completion.rstrip()


def infer_perturbation_name(dataset_source: str) -> str:
    return Path(dataset_source).stem if dataset_source else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run code generation on HumanEval")
    parser.add_argument("--dataset", default="data/HumanEval.jsonl")
    parser.add_argument("--prompt", default=None, help="single prompt file")
    parser.add_argument("--prompt-dir", default="prompts", help="directory of prompt files")    
    parser.add_argument("--model", default="dummy", choices=["dummy", "transformers"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--offline", action="store_true", help="use cached models only")
    parser.add_argument("--out", default="runs/codegen.jsonl")
    parser.add_argument("--invalid-log",default="runs/invalid_outputs.jsonl",help="where to append invalid (syntax) outputs as jsonl")
    args = parser.parse_args()
    
    

    invalid_log_path = Path(args.invalid_log)
    invalid_log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    prompt_files = []

    if args.prompt:
        prompt_files = [args.prompt]
    else:
        # load all .txt files in prompt directory
        prompt_files = sorted(
            str(p) for p in Path(args.prompt_dir).glob("*.txt")
        )

    if not prompt_files:
        raise SystemExit("No prompt files found.")

    print("Using prompt files:")
    for pf in prompt_files:
        print(" -", pf)
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
        tokenizer, model = load_transformers(args.model_path, device, args.offline)
        print(f"Using device: {device}")

    stop_strings = ["\n\ndef ", "\n\nclass ", "\n\n\n"]



    with out_path.open("w", encoding="utf-8") as out:
        for prompt_path in prompt_files:
            prompt_prefix = load_prompt(prompt_path)

            for idx, item in enumerate(load_humaneval(args.dataset, limit=args.limit), start=1):
                task_id = item.get("task_id")
                base_prompt = item.get("prompt", "")
                full_prompt = f"{prompt_prefix.rstrip()}\n\n{base_prompt.lstrip()}"

                for r in range(args.repeats):
                    print(f"[{Path(prompt_path).name}] Task {idx}/{args.limit} {task_id} repeat {r+1}/{args.repeats}")

                    if args.model == "dummy":
                        completion = dummy_complete(full_prompt)
                    else:
                        completion = generate_with_transformers(
                            tokenizer, model, full_prompt, args.max_new_tokens, device, stop_strings
                        )
                    code = extract_code_for_parse(completion)
                    valid_syntax, syntax_error = check_python_syntax(code)
                    completion = cleanup_completion(completion)

                    record = {
                        "task_id": task_id,
                        "prompt_file": Path(prompt_path).name,
                        "dataset_source": item.get("dataset_source", args.dataset),
                        "perturbation_name": infer_perturbation_name(item.get("dataset_source", args.dataset)),
                        "prompt": full_prompt,
                        "completion": completion,
                        "model": args.model,
                        "model_path": args.model_path,
                        "repeat": r,
                        "timestamp": time.time(),
                    }

                    #out.write(json.dumps(record) + "\n")
                    #out.flush()
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # write invalid log (extra)
                    if not valid_syntax:
                        with invalid_log_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
