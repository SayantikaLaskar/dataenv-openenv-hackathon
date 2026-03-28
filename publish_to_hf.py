from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update a Hugging Face Docker Space.")
    parser.add_argument("--repo-id", required=True, help="Space repo id, e.g. username/medical-triage-env")
    parser.add_argument("--folder", default=".", help="Folder to upload")
    args = parser.parse_args()

    token = (
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise SystemExit("Missing Hugging Face token in environment.")

    folder = Path(args.folder).resolve()
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=[".git/*", "__pycache__/*", "*.zip"],
    )
    print(f"https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
