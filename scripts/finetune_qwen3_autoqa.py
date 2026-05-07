"""
Fine-tune Qwen3 VLM for AutoQA.

Single CLI entry point that orchestrates:
  1. Stratified sampling of image pairs from vcape-r-20k and vcape-s-20k
  2. Formatting into Qwen3-VL chat-format training examples
  3. LoRA fine-tuning via PEFT + TRL SFTTrainer

Adapter weights are saved under <output-dir>/n<N>/ where N is the
number of samples per category, making it easy to compare performance
across different training set sizes.

Usage:
    python scripts/finetune_qwen3_autoqa.py --samples-per-category 1
    python scripts/finetune_qwen3_autoqa.py --samples-per-category 5 --epochs 10
    python scripts/finetune_qwen3_autoqa.py --samples-per-category 25 --resample --seed 123
    python scripts/finetune_qwen3_autoqa.py --vcape-r /path/to/vcape-r --vcape-s /path/to/vcape-s
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.finetune_sampler import sample_finetune_data
from scripts.training_formatter import format_all_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset paths (defaults — overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_VCAPE_R_PATH = "data/vcape-r-20k"
DEFAULT_VCAPE_S_PATH = "data/vcape-s-20k"
DEFAULT_SAMPLES_PATH = "data/finetune_samples"
DEFAULT_OUTPUT_DIR = "models/qwen3-autoqa-lora"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3 VLM for AutoQA",
    )
    parser.add_argument(
        "--vcape-r", type=str,
        default=DEFAULT_VCAPE_R_PATH,
        help="Path to the vcape-r-20k dataset on disk",
    )
    parser.add_argument(
        "--vcape-s", type=str,
        default=DEFAULT_VCAPE_S_PATH,
        help="Path to the vcape-s-20k dataset on disk",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model identifier for the base VLM",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for LoRA adapter weights (saved under n<N>/ sub-dir)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="Rank of the LoRA decomposition matrices",
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=32,
        help="Scaling factor for LoRA updates",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data sampling",
    )
    parser.add_argument(
        "--resample", action="store_true",
        help="Force re-sampling even if cached data exists",
    )
    parser.add_argument(
        "--samples-path", type=str,
        default=DEFAULT_SAMPLES_PATH,
        help="Root path to save/load the sampled dataset (saved under n<N>/ sub-dir)",
    )
    parser.add_argument(
        "--samples-per-category", type=int, default=1,
        help="Number of samples per rejection reason + accepted (default: 1)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    vcape_r_path = args.vcape_r
    vcape_s_path = args.vcape_s

    # Validate dataset paths
    for path, name in [(vcape_r_path, "vcape-r-20k"), (vcape_s_path, "vcape-s-20k")]:
        if not os.path.exists(path):
            print(
                f"Error: dataset path '{path}' ({name}) does not exist.",
                file=sys.stderr,
            )
            sys.exit(1)

    n_samples = args.samples_per_category

    # Store weights and samples under n<N>/ sub-directories so that
    # different sample counts never overwrite each other.
    output_dir = os.path.join(args.output_dir, f"n{n_samples}")
    samples_path = os.path.join(args.samples_path, f"n{n_samples}")

    # ------------------------------------------------------------------
    # Phase 1: Stratified sampling
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Phase 1: Stratified Data Sampling (N={n_samples} per category)")
    print(f"{'='*60}")

    ds = sample_finetune_data(
        vcape_r_path=vcape_r_path,
        vcape_s_path=vcape_s_path,
        output_path=samples_path,
        seed=args.seed,
        samples_per_category=n_samples,
        force=args.resample,
    )
    print(f"  Samples: {len(ds)} image pairs")

    # ------------------------------------------------------------------
    # Phase 2: Format training data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Phase 2: Training Data Formatting (N={n_samples})")
    print(f"{'='*60}")

    train_examples = format_all_examples(ds)
    print(f"  Formatted {len(train_examples)} training examples")

    # ------------------------------------------------------------------
    # Phase 3: Fine-tuning
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Phase 3: LoRA Fine-Tuning (N={n_samples})")
    print(f"{'='*60}")
    print(f"  Base model:  {args.base_model}")
    print(f"  LoRA rank:   {args.lora_rank}")
    print(f"  LoRA alpha:  {args.lora_alpha}")
    print(f"  LR:          {args.lr}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Output:      {output_dir}")

    # Lazy import to avoid loading torch/transformers when only sampling
    from scripts.fine_tuner import run_finetuning

    output_path = run_finetuning(
        train_dataset=train_examples,
        base_model=args.base_model,
        output_dir=output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Fine-Tuning Complete (N={n_samples})")
    print(f"{'='*60}")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Adapter saved to:  {output_path}")
    print(f"  Elapsed time:      {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
