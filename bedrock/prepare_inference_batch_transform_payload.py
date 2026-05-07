#!/usr/bin/env python3
"""
Prepare SageMaker Batch Transform Payloads from HuggingFace Dataset

Reads a HuggingFace-format dataset (load_from_disk), constructs vLLM-style
chat-completion payloads using the consistency evaluation prompt (matching
the evaluate_dataset.py framework), and uploads JSONL batch files to S3.

The payloads are formatted for the OpenAI-compatible chat API served by
the DJL/LMI + vLLM container (same format as evaluate_dataset.py autoqa).

Usage:
    # Prepare payloads from a HuggingFace dataset
    python prepare_inference_batch_transform_payload.py \
        --dataset ./full-dataset/vcape-r-20k \
        --s3-bucket your-s3-bucket \
        --s3-prefix batch_input_datasets/vcape-r-20k \
        --batch-size 500

    # With sample limits and shuffle
    python prepare_inference_batch_transform_payload.py \
        --dataset ./dataset \
        --s3-bucket my-bucket \
        --s3-prefix input/run-001 \
        --batch-size 1000 \
        --max-samples 5000 \
        --start-index 0 \
        --seed 42
"""

import argparse
import base64
import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from datasets import Dataset
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prepare_batch_transform_payload.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ===================================================================
#  Image helpers  (shared with evaluate_dataset.py)
# ===================================================================

def pil_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def half_resolution(img: Image.Image) -> Image.Image:
    """Downscale an image to half resolution."""
    img = img.convert("RGB")
    img.thumbnail((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    return img


# ===================================================================
#  Prompt  (same as evaluate_dataset.py autoqa)
# ===================================================================

SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, "
    "and the assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer."
)

PROMPT = (
    "\n**Task**: Determine if the generated object is CONSISTENT with the object in the source image.\n\n"
    "**Context**:\n"
    "- The SOURCE IMAGE shows the object in a lifestyle setting.\n"
    "- The ADDITIONAL SOURCE IMAGE shows the same object in a different setting.\n"
    "- The GENERATED image shows the same object isolated on a white background.\n\n"
    "Consistency is ensured when the material, texture, geometry, number of elements "
    "of the generated object are respected.\n\n"
    "**Consistency Criteria** - The following attributes MUST match:\n"
    "1. Core structural elements: Shape, form, and key geometric features\n"
    "2. Material and texture: Fabric type, wood, metal, leather, etc.\n"
    "3. Color: Primary colors and color distribution\n"
    "4. Orientation: The generated object must be oriented to the left. "
    "If the sofa is a corner sofa, the original orientation should be preserved.\n"
    "5. Quantity of essential structural components: Number of seats and cushions.\n\n"
    "These objects are always allowed in the generated image:\n"
    "- Decorative throw pillows, blankets, throws, plaids\n"
    "- Items inside sofa pockets/compartments or cupholders\n\n"
    "Conditionally allowed (ONLY if in original): Ottomans.\n\n"
    "Never allowed: People, pets, body parts, multiple main objects, "
    "lifestyle/cluttered environments, infographics/text/logos/watermarks, "
    "electronics (unless in pockets), unrelated objects.\n\n"
    "**Output Format**:\n"
    "<reasoning> ...your reasoning... </reasoning>'Yes' / 'No'"
)
#PROMPT = ( #QWEN 3.5 9B
#        "<prompt_text>\n<Task>Classify the consistency between the generated item and the source images as 'Yes' or 'No'.</Task>\n<Ground Truth>Ground-truth labels are absolutely correct. 'Yes' means consistent, 'No' means inconsistent.\n</Ground Truth>\n<Source Images>\n<source_image>Source image: Lifestyle setting with sofa.</source_image>\n<source_image>Additional source image: Same object in different setting.</source_image>\n<source_image>Generated image: Object isolated on white background.</source_image>\n</Source Images>\n<Consistency Rules>\nRule 1: Core Structure. Match shape, form, and key geometric features exactly. If any geometry artifacts exist in the generated image, classify as 'No'.\nRule 2: Material & Texture. Match fabric, wood, metal, and finish types exactly.\nRule 3: Color. Match primary colors and distribution exactly.\nRule 4: Orientation. Match orientation (e.g., left/right, corner). If corner orientation is changed, classify as 'No'.\nRule 5: Quantity. Match number of seats and cushions exactly.\nRule 6: Allowed Items. Decorative pillows, blankets, throws, and items inside pockets/compartments are allowed.\nRule 7: Conditional Items. Ottomans are allowed ONLY if present in the original source image.\nRule 8: Prohibited Items. People, pets, body parts, multiple main objects, lifestyle backgrounds, text, logs, watermarks, unrelated electronics (outside pockets) are forbidden.\nRule 9 (Priority): If the input metadata 'rejection_reason' contains 'Geometry Artifacts', 'Texture Mismatch', or 'Color Mismatch', the classification MUST be 'No', overriding visual evidence.\n</Consistency Rules>\n<Metadata Check>\nIf the 'rejection_reason' metadata is 'accepted', verify against visual evidence in Rules 1-8. If visual evidence contradicts acceptance (e.g., artifacts exist), classify as 'No'.\n</Metadata Check>\n<Output Format>\nOutput must be one of: '<reasoning> ... </reasoning>' followed by 'Yes' or 'No'.\nReasoning must explain why negative criteria were satisfied or violated. If 'No', explicitly mention the violated rule or rejection reason.\n</Output Format>\n</prompt_text>"
#    )

# ===================================================================
#  Payload construction
# ===================================================================

def build_chat_payload(
    sample: Dict,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
) -> Dict:
    """
    Build a vLLM OpenAI-compatible chat-completion payload from a
    HuggingFace dataset sample.

    Uses the same image/prompt structure as evaluate_dataset.py
    ``_make_autoqa_conversation``.

    The dataset is expected to have PIL Image columns:
      - ``source_image_main``
      - ``source_image_additional``
      - ``generated_image``
    """
    content_parts: List[Dict] = []

    image_fields = [
        ("SOURCE IMAGE", "source_image_main"),
        ("ADDITIONAL SOURCE IMAGE", "source_image_additional"),
        ("GENERATED IMAGE", "generated_image"),
    ]

    for label, key in image_fields:
        if key in sample and sample[key] is not None:
            img = half_resolution(sample[key])
            content_parts.append({"type": "text", "text": f"\n**{label}**"})
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{pil_to_base64(img)}"
                },
            })

    content_parts.append({"type": "text", "text": PROMPT})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]

    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": 1.5,
    }


def form_payload_row(record_id: str, sample: Dict, **kwargs) -> Dict:
    """
    Build a full JSONL row containing metadata and the inference payload.

    Matches the ``DataProcessing.InputFilter = $.payload`` convention used
    in ``launch_batch_transform_job.py``.
    """
    payload = build_chat_payload(sample, **kwargs)
    row = {
        "record_id": record_id,
        "payload": payload,
    }
    # Preserve ground-truth label if available (for later evaluation)
    if "is_generation_successful" in sample:
        row["ground_truth"] = sample["is_generation_successful"]
    return row


# ===================================================================
#  Batch creation & S3 upload
# ===================================================================

def create_and_upload_batches(
    dataset: Dataset,
    s3_bucket: str,
    s3_prefix: str,
    batch_size: int = 500,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
) -> List[Tuple[int, str]]:
    """
    Iterate over the dataset, build payloads, and upload JSONL batch
    files to S3.

    Returns:
        A list of ``(batch_index, s3_uri)`` tuples for every uploaded batch.
    """
    s3_client = boto3.client("s3")

    end_index = len(dataset) if max_samples is None else min(start_index + max_samples, len(dataset))
    total = end_index - start_index
    logger.info(
        "Creating batches: samples %d–%d (%d total), batch_size=%d",
        start_index, end_index - 1, total, batch_size,
    )

    uploaded_batches: List[Tuple[int, str]] = []
    batch_idx = 0
    current = start_index

    while current < end_index:
        batch_end = min(current + batch_size, end_index)
        payloads: List[str] = []

        for idx in tqdm(
            range(current, batch_end),
            desc=f"batch_{batch_idx} (samples {current}–{batch_end - 1})",
        ):
            sample = dataset[idx]
            record_id = f"sample_{idx}"
            row = form_payload_row(
                record_id, sample,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            payloads.append(json.dumps(row))

        jsonl_content = "\n".join(payloads)
        s3_key = f"{s3_prefix}/batch_{batch_idx}.jsonl"
        s3_uri = f"s3://{s3_bucket}/{s3_key}"

        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/jsonl",
        )
        logger.info("Uploaded %s (%d records)", s3_uri, len(payloads))
        uploaded_batches.append((batch_idx, s3_uri))

        batch_idx += 1
        current = batch_end

    logger.info("Total batches uploaded: %d", len(uploaded_batches))
    return uploaded_batches


# ===================================================================
#  Robust dataset loading  (same as evaluate_dataset.py)
# ===================================================================

def _patch_missing_shards(dataset_path: str) -> int:
    """
    Inspect state.json and remove references to missing arrow shard files.
    Returns the number of shards removed.
    """
    import os
    import shutil

    state_path = os.path.join(dataset_path, "state.json")
    if not os.path.isfile(state_path):
        return 0

    with open(state_path, "r") as f:
        state = json.load(f)

    data_files = state.get("_data_files", [])
    if not data_files:
        return 0

    existing, removed = [], 0
    for entry in data_files:
        fpath = os.path.join(dataset_path, entry["filename"])
        if os.path.isfile(fpath):
            existing.append(entry)
        else:
            logger.warning("Missing shard file, skipping: %s", fpath)
            removed += 1

    if removed > 0:
        if not existing:
            raise FileNotFoundError(
                f"All {len(data_files)} shard files are missing from {dataset_path}"
            )
        state["_data_files"] = existing
        backup_path = state_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(state_path, backup_path)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(
            "Patched state.json: removed %d missing shards, %d remaining",
            removed, len(existing),
        )

    return removed


def load_dataset_robust(dataset_path: str) -> Dataset:
    """Load a HuggingFace Dataset from disk, skipping missing shards."""
    removed = _patch_missing_shards(dataset_path)
    dataset = Dataset.load_from_disk(dataset_path)
    if removed > 0:
        logger.info(
            "Dataset loaded with %d shards skipped (%d samples available)",
            removed, len(dataset),
        )
    return dataset


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare SageMaker Batch Transform payloads from a HuggingFace dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Dataset
    p.add_argument(
        "--dataset", type=str, required=True,
        help="Path to a HuggingFace Dataset on disk (load_from_disk format).",
    )
    # S3 configuration
    p.add_argument(
        "--s3-bucket", type=str, required=True,
        help="S3 bucket for uploading JSONL batch files.",
    )
    p.add_argument(
        "--s3-prefix", type=str, required=True,
        help="S3 key prefix for batch files (e.g. batch_input_datasets/run-001).",
    )
    # Batching
    p.add_argument(
        "--batch-size", type=int, default=500,
        help="Number of samples per JSONL batch file (default: 500).",
    )
    p.add_argument(
        "--start-index", type=int, default=0,
        help="Start index in the dataset (default: 0).",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Max number of samples to process (default: all).",
    )
    # Model inference parameters
    p.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Max tokens for model generation (default: 1024).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8).",
    )
    p.add_argument(
        "--top-p", type=float, default=0.8,
        help="Top-p sampling (default: 0.8).",
    )
    # Misc
    p.add_argument(
        "--seed", type=int, default=None,
        help="Shuffle dataset with this seed before batching.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Load dataset ------------------------------------------------
    logger.info("Loading dataset from: %s", args.dataset)
    dataset = load_dataset_robust(args.dataset)
    logger.info("Dataset loaded: %d samples, columns: %s", len(dataset), dataset.column_names)

    if args.seed is not None:
        dataset = dataset.shuffle(seed=args.seed)
        logger.info("Dataset shuffled with seed=%d", args.seed)

    # ---- Create and upload batches -----------------------------------
    logger.info("=" * 60)
    logger.info("Preparing Batch Transform Payloads")
    logger.info("=" * 60)
    logger.info("  S3 Bucket   : %s", args.s3_bucket)
    logger.info("  S3 Prefix   : %s", args.s3_prefix)
    logger.info("  Batch Size  : %d", args.batch_size)
    logger.info("  Start Index : %d", args.start_index)
    logger.info("  Max Samples : %s", args.max_samples or "all")
    logger.info("  Max Tokens  : %d", args.max_tokens)
    logger.info("  Temperature : %.2f", args.temperature)
    logger.info("  Top-p       : %.2f", args.top_p)

    uploaded = create_and_upload_batches(
        dataset=dataset,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        batch_size=args.batch_size,
        start_index=args.start_index,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    logger.info("✓ Payload preparation complete! %d batches uploaded.", len(uploaded))
    for batch_idx, s3_uri in uploaded:
        logger.info("  Batch %d: %s", batch_idx, s3_uri)


if __name__ == "__main__":
    main()
