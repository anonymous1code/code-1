"""
Evaluate a fine-tuned Qwen3-VL LoRA adapter on held-out VCaPE data.

Loads LoRA weights, excludes training rows via the saved manifest,
and evaluates on the remaining test set.

Usage:
    python scripts/infer_qwen3_autoqa.py \
        --adapter-path models/qwen3-autoqa-lora/n5 \
        --dataset vcape-r

    python scripts/infer_qwen3_autoqa.py \
        --adapter-path models/qwen3-autoqa-lora/n5 \
        --dataset vcape-s \
        --output results/n5_vcape_s.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.training_formatter import EVAL_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DATASET_PATHS = {
    "vcape-r": ("data/vcape-r-20k", "vcape-r-20k"),
    "vcape-s": ("data/vcape-s-20k", "vcape-s-20k"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(adapter_path: str):
    """Load base model + LoRA adapter, merge, return (model, processor)."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path) as f:
        base_model = json.load(f).get("base_model_name_or_path",
                                       "Qwen/Qwen3-VL-8B-Instruct")

    logger.info("Base model: %s", base_model)
    device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model)
    return model, processor


def load_train_indices(adapter_path: str, manifest_key: str) -> set[int]:
    """Read train_indices.json from the matching finetune_samples dir."""
    n_dir = os.path.basename(adapter_path.rstrip("/"))          # e.g. "n5"
    manifest = os.path.join("data", "finetune_samples", n_dir, "train_indices.json")
    if not os.path.isfile(manifest):
        logger.warning("No manifest at %s — no rows excluded", manifest)
        return set()
    with open(manifest) as f:
        data = json.load(f)
    indices = data.get("all_indices", {}).get(manifest_key, [])
    return set(indices)


def predict(model, processor, source, target, max_tokens: int = 1024) -> str:
    """Run the consistency prompt and return 'Yes', 'No', or 'Unknown'."""
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": source},
            {"type": "image", "image": target},
            {"type": "text", "text": EVAL_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    imgs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, padding=True,
                       return_tensors="pt").to(model.device)

    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_tokens)
    raw = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].strip()

    # Extract final Yes/No after </reasoning>, fallback to anywhere
    m = re.search(r"</reasoning>\s*(.*)", raw, re.DOTALL)
    tail = m.group(1) if m else raw
    hit = re.search(r"\b(Yes|No)\b", tail)
    return hit.group(1) if hit else "Unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen3-VL on held-out data.")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--dataset", required=True, choices=["vcape-r", "vcape-s"])
    ap.add_argument("--output", default=None, help="Path to save results JSON.")
    args = ap.parse_args()

    ds_path, manifest_key = _DATASET_PATHS[args.dataset]

    # 1. Load model
    print(f"Loading adapter from {args.adapter_path} ...")
    model, processor = load_model(args.adapter_path)

    # 2. Load dataset and exclude training rows
    train_idx = load_train_indices(args.adapter_path, manifest_key)
    ds = load_from_disk(ds_path)
    keep = [i for i in range(len(ds)) if i not in train_idx]
    ds = ds.select(keep)
    print(f"Dataset: {args.dataset}  total={len(keep)+len(train_idx)}  "
          f"train_excluded={len(train_idx)}  eval={len(ds)}")

    # 3. Evaluate
    correct = 0
    results = []
    t0 = time.time()

    for i in range(len(ds)):
        row = ds[i]
        gt = "Yes" if row["label"].strip().lower() in {"accepted", "accept"} else "No"
        pred = predict(model, processor,
                       row["xsource_image"].convert("RGB"),
                       row["xtarget_image"].convert("RGB"))
        ok = pred == gt
        correct += int(ok)
        results.append({
            "ground_truth": gt,
            "predicted": pred,
            "correct": ok,
            "rejection_reason": row.get("rejection_reason", ""),
        })
        print(f"  [{i+1:5d}/{len(ds)}] {'✓' if ok else '✗'}  GT={gt}  Pred={pred}  "
              f"{row.get('rejection_reason', '')[:50]}")

    elapsed = time.time() - t0
    acc = correct / len(ds)
    print(f"\nAccuracy: {correct}/{len(ds)} = {acc:.1%}  ({elapsed:.0f}s)")

    # 4. Save
    out = args.output or f"results/infer_{args.dataset}_{os.path.basename(args.adapter_path)}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"accuracy": acc, "correct": correct, "total": len(ds),
                    "elapsed_s": round(elapsed, 1), "predictions": results},
                  f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
