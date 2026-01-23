import argparse
import shutil
import sys
from pathlib import Path
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


def convert_checkpoint_to_safetensors(
    input_pth: Path,
    output_dir: Path,
):
    """
    Uses convert_to_safetensors.py exactly as provided (CLI invocation).
    Produces model.safetensors in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "model.safetensors"

    script_path = Path(__file__).resolve().parent / "convert_to_safetensors.py"
    if not script_path.exists():
        raise FileNotFoundError(f"convert_to_safetensors.py not found at {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        str(input_pth),
        str(out_file),
    ]

    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not out_file.exists():
        raise RuntimeError("model.safetensors was not created")

    print("[INFO] Safetensors saved to:", out_file)


def copy_qwen3gdn_code(repo_root: Path, output_dir: Path):
    """
    Copies contents of qwen3gdn into output_dir without overwriting existing files
    """
    src = repo_root / "qwen3gdn"

    if not src.exists():
        raise FileNotFoundError(f"Missing qwen3gdn directory at {src}")

    print(f"[INFO] Copying HF code from {src} → {output_dir}")

    for item in src.iterdir():
        dst = output_dir / item.name

        if item.is_dir():
            if dst.exists():
                print(f"[INFO] Skipping existing directory: {dst.name}")
            else:
                shutil.copytree(item, dst)
        else:
            if dst.exists():
                print(f"[INFO] Skipping existing file: {dst.name}")
            else:
                shutil.copy2(item, dst)



def save_tokenizer(base_model: str, output_dir: Path):
    """
    Downloads tokenizer from base_model and saves locally
    """
    print(f"[INFO] Downloading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_dir)


def verify_hf_layout(output_dir: Path):
    """
    Minimal sanity checks
    """
    required = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
    ]

    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        print("[WARN] Missing expected HF files:", missing)
    else:
        print("[INFO] HF export looks complete ✅")


def main():
    parser = argparse.ArgumentParser("Export RADLADS/Qwen3 GDN to HF format")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)

    args = parser.parse_args()

    input_pth = Path(args.input_path).resolve()
    output_dir = Path(args.output_path).resolve()
    repo_root = Path(__file__).resolve().parent

    if not input_pth.exists():
        raise FileNotFoundError(input_pth)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert checkpoint → safetensors
    convert_checkpoint_to_safetensors(input_pth, output_dir)

    # 2. Save tokenizer
    save_tokenizer(args.base_model, output_dir)

    # 3. Copy HF modeling code
    copy_qwen3gdn_code(repo_root, output_dir)

    # 4. Final sanity check
    verify_hf_layout(output_dir)

    print("\n[SUCCESS] HF-compatible checkpoint exported 🎉")
    print(f"→ {output_dir}")


if __name__ == "__main__":
    main()