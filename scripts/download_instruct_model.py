import argparse
import os
from pathlib import Path
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an instruct model to a local folder")
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--out",
        default="models/qwen2.5-coder-1.5b-instruct",
        help="Local output directory",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision/branch/tag/commit",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional include pattern (repeatable), e.g. '*.safetensors'",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional exclude pattern (repeatable), e.g. '*.bin'",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file containing HF token variables",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    load_dotenv(args.env_file)
    token = os.getenv("HFTOKEN")
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        token=token,
    )
    print(f"Downloaded model to: {path}")


if __name__ == "__main__":
    main()
