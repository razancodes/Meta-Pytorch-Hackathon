#!/usr/bin/env python3
"""
Memex OS-Agent — LoRA Hot-Swap Utility.

Dynamically reload updated LoRA adapter weights into a running model
without restarting the base model or reloading it from disk. This enables
zero-downtime continuous learning: DPO training produces new adapters,
and this utility swaps them into the live inference server.

Usage:
  # Standalone test
  python hotswap.py --base unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \\
      --adapter checkpoints/dpo-latest

  # Programmatic (import into inference.py)
  from hotswap import hot_reload_lora
  success = hot_reload_lora(model, "checkpoints/dpo-latest")
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def hot_reload_lora(
    model: torch.nn.Module,
    adapter_path: str,
    adapter_name: str = "default",
    device: Optional[str] = None,
) -> bool:
    """Reload LoRA adapter weights from disk into a running model.

    This swaps ONLY the trainable LoRA parameters (A/B matrices),
    leaving the frozen base model weights untouched. No model reload,
    no VRAM spike, no downtime.

    Args:
        model: A PEFT-wrapped model (from Unsloth or HuggingFace PEFT).
        adapter_path: Path to directory containing adapter_model.safetensors
                      or adapter_model.bin (output of model.save_pretrained()).
        adapter_name: Name of the adapter to update (default: "default").
        device: Target device. If None, inferred from model parameters.

    Returns:
        True if successful, False otherwise.
    """
    if device is None:
        device = next(model.parameters()).device

    # Locate the adapter weights file
    safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        adapter_state = load_file(safetensors_path, device=str(device))
        source = "safetensors"
    elif os.path.exists(bin_path):
        adapter_state = torch.load(bin_path, map_location=device, weights_only=True)
        source = "bin"
    else:
        print(f"  ❌ No adapter weights found at: {adapter_path}")
        print(f"     Expected: adapter_model.safetensors or adapter_model.bin")
        return False

    # Load the state dict into the model
    try:
        # Method 1: PEFT's native load_adapter (preferred)
        if hasattr(model, "load_adapter"):
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            model.set_adapter(adapter_name)
            print(f"  ✓ Hot-swapped via load_adapter ({source})")
            return True
    except Exception as e:
        print(f"  ⚠ load_adapter failed: {e}, falling back to manual swap")

    # Method 2: Manual state dict injection
    model_state = model.state_dict()
    swapped = 0
    skipped = 0

    for key, new_tensor in adapter_state.items():
        # Adapter keys may have different prefixes; try common patterns
        candidates = [key]
        if not key.startswith("base_model."):
            candidates.append(f"base_model.model.{key}")
            candidates.append(f"base_model.{key}")

        matched = False
        for candidate in candidates:
            if candidate in model_state:
                if model_state[candidate].shape == new_tensor.shape:
                    model_state[candidate].copy_(new_tensor.to(device))
                    swapped += 1
                    matched = True
                    break
                else:
                    print(
                        f"  ⚠ Shape mismatch: {candidate} "
                        f"model={model_state[candidate].shape} "
                        f"adapter={new_tensor.shape}"
                    )

        if not matched:
            skipped += 1

    if swapped == 0:
        print(f"  ❌ No parameters were swapped (skipped {skipped})")
        return False

    print(f"  ✓ Hot-swapped {swapped} parameters ({skipped} skipped) [{source}]")
    return True


def get_adapter_info(adapter_path: str) -> dict:
    """Get metadata about a LoRA adapter checkpoint."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    info = {
        "path": adapter_path,
        "exists": os.path.isdir(adapter_path),
        "has_safetensors": os.path.exists(
            os.path.join(adapter_path, "adapter_model.safetensors")
        ),
        "has_bin": os.path.exists(
            os.path.join(adapter_path, "adapter_model.bin")
        ),
        "has_config": os.path.exists(config_path),
    }

    if info["has_config"]:
        import json
        with open(config_path) as f:
            config = json.load(f)
        info["r"] = config.get("r")
        info["lora_alpha"] = config.get("lora_alpha")
        info["target_modules"] = config.get("target_modules")

    # File size
    for name in ["adapter_model.safetensors", "adapter_model.bin"]:
        fp = os.path.join(adapter_path, name)
        if os.path.exists(fp):
            info["size_mb"] = os.path.getsize(fp) / (1024 * 1024)
            break

    return info


# ═══════════════════════════════════════════════════════════════════════
# CLI — Standalone test
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="LoRA Hot-Swap Utility")
    p.add_argument("--base", required=True, help="Base model name or path")
    p.add_argument("--adapter", required=True, help="Path to new adapter weights")
    p.add_argument("--info-only", action="store_true", help="Just print adapter info")
    args = p.parse_args()

    if args.info_only:
        info = get_adapter_info(args.adapter)
        for k, v in info.items():
            print(f"  {k}: {v}")
        return

    print(f"\n{'═'*60}")
    print(f"  LORA HOT-SWAP TEST")
    print(f"  Base:    {args.base}")
    print(f"  Adapter: {args.adapter}")
    print(f"{'═'*60}\n")

    # Load base model
    print("[1/2] Loading base model...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=2048, load_in_4bit=True,
    )
    print("  ✓ Loaded")

    # Hot-swap
    print("[2/2] Hot-swapping adapters...")
    t0 = time.time()
    success = hot_reload_lora(model, args.adapter)
    elapsed = time.time() - t0

    if success:
        print(f"\n  ✅ Hot-swap complete in {elapsed:.2f}s")
    else:
        print(f"\n  ❌ Hot-swap failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
