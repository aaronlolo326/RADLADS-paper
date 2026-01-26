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


def copy_qwen3gdn_code(repo_root: Path, output_dir: Path, config_dir: Path):
    """
    Copies contents of qwen3gdn into output_dir without overwriting existing files
    """
    src = repo_root / "qwen3gdn"

    if not src.exists():
        raise FileNotFoundError(f"Missing qwen3gdn directory at {src}")

    print(f"[INFO] Copying HF code from {src} → {output_dir}")

    import json
    import shutil
    import yaml

    for item in src.iterdir():
        dst = output_dir / item.name

        if item.is_dir():
            if dst.exists() and False:
                print(f"[INFO] Skipping existing directory: {dst.name}")
            else:
                print(f"[INFO] Copying tree: {item} -> {dst.name}")
                shutil.copytree(item, dst)
        else:
            if dst.exists() and False:
                print(f"[INFO] Skipping existing file: {dst.name}")
            else:
                if item.name == "config.json":
                    # Special processing for config.json using [config_dir]/gdn.yaml's preserve_layers_lst
                    gdn_yaml_path = config_dir / "gdn.yaml"
                    if not gdn_yaml_path.exists():
                        raise FileNotFoundError(f"gdn.yaml not found in config_dir: {gdn_yaml_path}")
                    with open(gdn_yaml_path, "r") as f:
                        gdn_yaml = yaml.safe_load(f)
                    preserve_layers_lst = []
                    if isinstance(gdn_yaml, dict):
                        # Try to access nested under model: key if present
                        if "model" in gdn_yaml and isinstance(gdn_yaml["model"], dict):
                            preserve_layers_lst = gdn_yaml["model"].get("preserve_layers_lst", [])
                        else:
                            preserve_layers_lst = gdn_yaml.get("preserve_layers_lst", [])
                    # Default to empty list if parsing failed

                    with open(item, "r") as f:
                        config_data = json.load(f)

                    num_hidden_layers = config_data.get("num_hidden_layers", None)
                    # Compose layer_types from preserve_layers_lst (1->"full_attention", 0->"linear_attention")
                    if num_hidden_layers is not None:
                        layer_types = []
                        preserve_set = set(preserve_layers_lst if preserve_layers_lst is not None else [])
                        for i in range(num_hidden_layers):
                            if i in preserve_set:
                                layer_types.append("full_attention")
                            else:
                                layer_types.append("linear_attention")
                        config_data["layer_types"] = layer_types

                        # Write the modified config.json to its destination
                        with open(dst, "w") as f_out:
                            json.dump(config_data, f_out, indent=2)
                        print(f"[INFO] Wrote patched config.json to {dst}")
                    else:
                        # Fallback: straight copy if num_hidden_layers missing
                        shutil.copy2(item, dst)
                        print(f"[WARN] num_hidden_layers not found in config.json, copied as-is to {dst}")
                else:
                    shutil.copy2(item, dst)
                    print(f"[INFO] Copying: {item} -> {dst.name}")



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
    parser.add_argument("--config_dir", type=str, required=True)

    args = parser.parse_args()

    input_pth = Path(args.input_path).resolve()
    output_dir = Path(args.output_path).resolve()
    config_dir = Path(args.config_dir).resolve()
    repo_root = Path(__file__).resolve().parent

    if not input_pth.exists():
        raise FileNotFoundError(input_pth)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert checkpoint → safetensors
    convert_checkpoint_to_safetensors(input_pth, output_dir)

    # 2. Save tokenizer
    save_tokenizer(args.base_model, output_dir)

    # 3. Copy HF modeling code
    copy_qwen3gdn_code(repo_root, output_dir, config_dir)

    # 4. Final sanity check
    verify_hf_layout(output_dir)

    print("\n[SUCCESS] HF-compatible checkpoint exported 🎉")
    print(f"→ {output_dir}")


if __name__ == "__main__":
    main()