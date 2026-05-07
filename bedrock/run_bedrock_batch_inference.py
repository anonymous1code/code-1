#!/usr/bin/env python3
"""
Bedrock Batch Inference for Full-Dataset (HuggingFace format)

Reads HuggingFace-format datasets (xsource_image, xtarget_image, etc.),
constructs Bedrock batch payloads using the consistency evaluation prompt
from bedrock_evaluator.py, uploads JSONL batches to S3, and optionally
submits batch inference jobs.

Usage:
    # Prepare batches only (upload JSONL to S3)
    python run_bedrock_batch_inference.py \
        --dataset ../../full-dataset/vcape-r-20k \
        --s3-bucket your-s3-bucket \
        --s3-prefix bedrock_input_datasets/vcape-r-20k \
        --batch-size 500

    # Prepare batches and submit jobs
    python run_bedrock_batch_inference.py \
        --dataset ../../full-dataset/vcape-r-20k \
        --s3-bucket your-s3-bucket \
        --s3-prefix bedrock_input_datasets/vcape-r-20k \
        --batch-size 500 \
        --submit-jobs \
        --role-arn arn:aws:iam::123456789012:role/YourBedrockBatchRole \
        --model-id us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
        --job-name-prefix vcape-r-20k
"""

import argparse
import base64
import io
import json
import logging
import sys
from enum import Enum
from typing import Optional
from datetime import datetime
import boto3
import pandas as pd
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from botocore.exceptions import ClientError
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("run_bedrock_batch_inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OutputLabel (mirrors bedrock_evaluator.py)
# ---------------------------------------------------------------------------
class OutputLabel(Enum):
    """Valid output labels - single source of truth."""
    ACCEPTABLE = ("acceptable", "The generated sofa is consistent.")
    GEOMETRY_ARTIFACTS = ("geometry_artifacts", "Missing seats, wrong shape, incorrect dimensions. Artifacts introduced in the side of the sofa that are not present in the original sofa.")
    TEXTURE_LIGHTING_COLOR_ISSUES = ("texture_lighting_color_issues", "Wrong material, color mismatch, lighting problems.")
    WRONG_ORIENTATION = ("wrong_orientation", "Object facing wrong direction.")
    BACKGROUND_ISSUES = ("background_issues", "Non-white background, environmental elements present.")
    IRRELEVANT_OBJECTS = ("irrelevant_objects", "Forbidden objects present (people, pets, multiple sofas, etc.)")
    NON_RELEVANT_MAIN_IMAGE = ("non_relevant_main_image", "The additional source image is non relevant.")

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def description(self) -> str:
        return self.value[1]

    @classmethod
    def get_prompt_description(cls) -> str:
        """Generate the label list for the prompt."""
        lines = ["**Output Labels List** (use the exact label, and only one):"]
        for item in cls:
            lines.append(f"- '{item.label}': {item.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt (from bedrock_evaluator.py / evaluate_dataset.py)
# Adapted for 2-image case (source + generated, no additional source)
# ---------------------------------------------------------------------------
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

PROMPT = "**Task**: Determine if the generated object is CONSISTENT with the object in the source image.\n\n**Context**:\n- The SOURCE IMAGE shows the object in a lifestyle setting.\n- The ADDITIONAL SOURCE IMAGE shows the same object in a different setting.\n- The GENERATED image shows the same object isolated on a white background.\n\n**Critical Instruction**: You MUST identify and articulate specific mismatches. Consistency means the generated object is the EXACT SAME PRODUCT, not merely similar or the same category.\n\n**Mandatory Rejection Criteria** - Mark as \"No\" if ANY of these apply:\n\n1. **Material Mismatch**: \n   - Fabric vs. leather difference\n   - Different fabric textures (linen vs. velvet vs. woven)\n   - Different leather finishes (smooth vs. distressed)\n   - Wood vs. metal vs. plastic differences\n   - Any visible material difference = INCONSISTENT\n\n2. **Color Discrepancy**:\n   - Different shade families (beige vs. gray, navy vs. black)\n   - Significantly lighter or darker tones\n   - Different color patterns or distributions\n   - If you can describe the colors with different words = INCONSISTENT\n\n3. **Structural Count Mismatch**:\n   - Different number of seats (2-seater vs. 3-seater)\n   - Different number of cushions or sections\n   - Different number of legs, arms, or support structures\n   - Different number of light bulbs, shades, or fixture arms\n   - Any countable difference = INCONSISTENT\n\n4. **Configuration Difference**:\n   - Different furniture arrangement (straight vs. L-shaped vs. sectional)\n   - Different proportions (compact vs. oversized)\n   - Different design style (modern vs. traditional vs. industrial)\n   - Different form factor = INCONSISTENT\n\n5. **Design Feature Mismatch**:\n   - Presence/absence of tufting, stitching patterns, or quilting\n   - Different armrest styles (rolled vs. square vs. track)\n   - Different leg designs (tapered vs. straight vs. hairpin)\n   - Different backrest styles (high-back vs. low-back vs. backless)\n   - Any distinctive design difference = INCONSISTENT\n\n6. **Prohibited Elements Present**:\n   - People, pets, hands, or body parts\n   - Multiple distinct objects of the same furniture type\n   - Environment elements (walls, floors, plants, decor)\n   - Text, logos, watermarks, or graphics\n   - Any prohibited element = INCONSISTENT\n\n**Acceptable Elements** (DO NOT use these as reasons for inconsistency):\n- Decorative pillows or throws placed on the object\n- Items in pockets or cupholders\n- White background vs. lifestyle background\n- Slight angle differences in product photography\n\n**Evaluation Protocol**:\n\nStep 1: Examine the source images and describe the object precisely:\n- Material type\n- Primary color(s) and shades\n- Count of seats/sections/components\n- Configuration and shape\n- Key design features\n\nStep 2: Examine the generated image and describe it using the same attributes\n\nStep 3: Compare descriptions point-by-point:\n- List ANY differences you observe\n- If you find even ONE substantive difference from criteria 1-6, mark as \"No\"\n\nStep 4: Check for prohibited elements in generated image\n\nStep 5: Make final determination:\n- \"Yes\" ONLY if descriptions match on ALL attributes and no prohibited elements\n- \"No\" if ANY mismatch exists or ANY prohibited element is present\n\n**Scoring Principle**: When in doubt between similar products, mark as \"No\". The task requires identifying the EXACT SAME product, not a similar alternative.\n\n**Output Format**:\nThis is a classification task with labels [\"Yes\", \"No\"].\n<reasoning>\n[Step 1: Source object description]\n[Step 2: Generated object description]\n[Step 3: Point-by-point comparison with specific differences noted]\n[Step 4: Prohibited elements check]\n[Step 5: Final determination with justification]\n</reasoning>\n'Yes' / 'No'"
# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def pil_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buf = io.BytesIO()
    img = image.convert("RGB")
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def half_resolution(img: Image.Image) -> Image.Image:
    """Downscale an image to half resolution."""
    img = img.convert("RGB")
    img.thumbnail((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    return img


def create_presigned_url(bucket_name, object_key, expiration=3600):
    """
    Generate a presigned URL to share an S3 object.

    :param bucket_name: Name of the S3 bucket.
    :param object_key: Key (path) of the object in the bucket.
    :param expiration: Time in seconds for the URL to remain valid (default: 1 hour).
    :return: Presigned URL as a string, or None on error.
    """
    s3_client = boto3.client("s3")

    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None

    return url

# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def form_payload(record_id: str, sample: dict, model_id: str) -> dict:
    """
    Build a single Bedrock batch inference payload from a full-dataset sample.

    The full-dataset has:
      - xsource_image  (PIL Image) → SOURCE IMAGE
      - xtarget_image  (PIL Image) → GENERATED IMAGE

    We downscale the source image to half-resolution (same as evaluate_dataset.py)
    but keep the generated image at full resolution.
    """
    content = []

    if model_id == "" or "claude" in model_id:
        # Source image (half-res like evaluate_dataset)
        source_img = half_resolution(sample["xsource_image"])
        content.append({"type": "text", "text": "[Source Image Main]"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": pil_to_base64(source_img),
            },
        })

        # Generated / target image (full resolution)
        content.append({"type": "text", "text": "[Generated Image]"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": pil_to_base64(sample["xtarget_image"]),
            },
        })

        # Prompt
        content.append({"type": "text", "text": PROMPT})
    elif "nova" in model_id or "qwen" in model_id:
         # Source image (half-res like evaluate_dataset)
        source_img = half_resolution(sample["xsource_image"])
        content.append({ "text": "[Source Image Main]"})
        content.append({
            "image": {
                "format": "jpeg",
                "source": {
                    "bytes": pil_to_base64(source_img),
                }
            },
        })

        # Generated / target image (full resolution)
        content.append({"text": "[Generated Image]"})
        content.append({
            "image": {
                "format": "jpeg",
                "source": {
                    "bytes":pil_to_base64(sample["xtarget_image"]),
                }
            },
        })

        # Prompt
        content.append({"text": PROMPT})
    elif "meta" in model_id:
        source_img = half_resolution(sample["xsource_image"])
        source_b64 = pil_to_base64(source_img)
        target_b64 = pil_to_base64(sample["xtarget_image"])

        # Llama4 Maverick on Bedrock batch inference uses the native Llama
        # format with `prompt` key. For multimodal inputs, images are
        # passed as base64 content within special image tags in the prompt.
        # Format: <|image|><|begin_of_text|>base64_data<|end_of_text|>
        #
        # Alternatively, Llama4 on Bedrock supports a chat-completion style
        # with `prompt` containing the full formatted conversation.
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"[Source Image Main]\n<|image|>{source_b64}\n"
            f"[Generated Image]\n<|image|>{target_b64}\n"
            f"{PROMPT}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        return {
            "recordId": record_id,
            "modelInput": {
                "prompt": prompt,
                "max_gen_len": 20000,
            },
        }
    elif "mistral" in model_id:
        source_img = half_resolution(sample["xsource_image"])
        formatted_prompt = f"""<s>[INST] 
[Source Image Main]: {pil_to_base64(source_img)},
[Generated Image]: {pil_to_base64(sample["xtarget_image"])}
{PROMPT} [/INST]"""

        return {
            "recordId": record_id,
            "modelInput": {
                "prompt": formatted_prompt,
                "max_tokens": 20000,
            },
        }

    # Default: Anthropic Claude format
    return {
        "recordId": record_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20000,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        },
    }
    


# ---------------------------------------------------------------------------
# Batch creation & S3 upload
# ---------------------------------------------------------------------------

def _ensure_pil_image(img) -> Image.Image:
    """Convert a value from a pandas cell back to a PIL Image.

    HuggingFace ``Dataset.to_pandas()`` serialises Image columns as dicts
    (e.g. ``{"bytes": b"...", "path": "..."}``).  This helper transparently
    handles both that dict format and already-decoded PIL Images.
    """
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict):
        if "bytes" in img and img["bytes"]:
            return Image.open(io.BytesIO(img["bytes"]))
        if "path" in img and img["path"]:
            return Image.open(img["path"])
    raise TypeError(f"Cannot convert {type(img)} to PIL Image")


def _row_to_jsonl(row: pd.Series, model_id: str) -> str:
    """Convert a single DataFrame row to a JSON-serialised payload string.

    This is the function passed to ``DataFrame.apply`` so that payload
    construction is expressed as a vectorised-style operation rather than
    a manual ``for`` loop.
    """
    sample = {
        "xsource_image": _ensure_pil_image(row["xsource_image"]),
        "xtarget_image": _ensure_pil_image(row["xtarget_image"]),
    }
    payload = form_payload(row["record_id"], sample, model_id)
    return json.dumps(payload)


def create_and_upload_batches(
    dataset: Dataset,
    s3_bucket: str,
    s3_prefix: str,
    batch_size: int = 500,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    model_id: str = ""
) -> list:
    """
    Convert the dataset slice to a pandas DataFrame, use ``apply`` to build
    JSONL payloads, then upload each batch to S3.

    Returns a list of (batch_index, s3_uri) tuples.
    """
    s3_client = boto3.client("s3")

    end_index = len(dataset) if max_samples is None else min(start_index + max_samples, len(dataset))
    total = end_index - start_index
    logger.info(
        "Creating batches: samples %d–%d (%d total), batch_size=%d",
        start_index, end_index - 1, total, batch_size,
    )

    # --- Slice and convert to pandas ----------------------------------
    subset = dataset.select(range(start_index, end_index))
    # keep image columns decoded (PIL) by setting format to None (default)
    df = subset.to_pandas()

    # Add helper columns
    df["record_id"] = [f"sample_{i}" for i in range(start_index, end_index)]
    df["batch_idx"] = [i // batch_size for i in range(total)]

    # --- Build payloads with apply ------------------------------------
    logger.info("Building payloads with pandas apply …")
    tqdm.pandas(desc="Forming payloads")
    df["jsonl"] = df.progress_apply(lambda row: _row_to_jsonl(row, model_id), axis=1)

    # --- Group by batch and upload ------------------------------------
    uploaded_batches = []
    for batch_id, group in df.groupby("batch_idx", sort=True):
        jsonl_content = "\n".join(group["jsonl"])
        s3_key = f"{s3_prefix}/batch_{batch_id}.jsonl"
        s3_uri = f"s3://{s3_bucket}/{s3_key}"

        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=jsonl_content.encode("utf-8"),
            ContentType="application/jsonl",
        )
        logger.info("Uploaded %s (%d records)", s3_uri, len(group))
        uploaded_batches.append((batch_id, s3_uri))

    logger.info("Total batches uploaded: %d", len(uploaded_batches))
    return uploaded_batches


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_batch_jobs(
    s3_bucket: str,
    s3_prefix: str,
    s3_output_prefix: str,
    num_batches: int,
    role_arn: str,
    model_id: str,
    job_name_prefix: str,
    batch_start: int = 0,
    batch_end: Optional[int] = None,
):
    """Submit Bedrock batch inference jobs for uploaded batch files."""
    bedrock = boto3.client(service_name="bedrock", region_name="us-east-1")

    if batch_end is None:
        batch_end = num_batches
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for batch_idx in range(batch_start, batch_end):
        input_s3_uri = f"s3://{s3_bucket}/{s3_prefix}/batch_{batch_idx}.jsonl"
        output_s3_uri = f"s3://{s3_bucket}/{s3_output_prefix}/batch_{batch_idx}/"
        job_name = f"{job_name_prefix}-batch-{batch_idx}-{timestamp}"

        logger.info("Submitting job: %s", job_name)
        logger.info("  Input:  %s", input_s3_uri)
        logger.info("  Output: %s", output_s3_uri)

        response = bedrock.create_model_invocation_job(
            roleArn=role_arn,
            modelId=model_id,
            jobName=job_name,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": input_s3_uri,
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": output_s3_uri,
                }
            },
        )

        job_arn = response.get("jobArn")
        logger.info("  JobArn: %s", job_arn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Bedrock Batch Inference for full-dataset (HuggingFace format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Dataset
    p.add_argument(
        "--dataset", type=str, required=True,
        help="Path to HuggingFace dataset on disk (load_from_disk format).",
    )
    # S3 configuration
    p.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket for input/output.")
    p.add_argument("--s3-prefix", type=str, required=True, help="S3 key prefix for input JSONL batches.")
    p.add_argument(
        "--s3-output-prefix", type=str, default=None,
        help="S3 key prefix for output. Defaults to s3-prefix with 'input' replaced by 'output'.",
    )
    # Batching
    p.add_argument("--batch-size", type=int, default=500, help="Number of samples per JSONL batch file.")
    p.add_argument("--start-index", type=int, default=0, help="Start index in the dataset.")
    p.add_argument("--max-samples", type=int, default=None, help="Max number of samples to process.")
    # Job submission
    p.add_argument("--submit-jobs", action="store_true", help="Submit Bedrock batch inference jobs after uploading.")
    p.add_argument(
        "--role-arn", type=str, default="arn:aws:iam::123456789012:role/YourBedrockBatchRole",
        help="IAM role ARN for Bedrock batch jobs.",
    )
    p.add_argument(
        "--model-id", type=str,
        default="arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        help="Bedrock model ID or inference profile ARN.",
    )
    p.add_argument("--job-name-prefix", type=str, default="batch-inference", help="Prefix for batch job names.")
    # Submission range (for submitting a subset of already-uploaded batches)
    p.add_argument("--batch-start", type=int, default=0, help="First batch index to submit (for --submit-jobs).")
    p.add_argument("--batch-end", type=int, default=None, help="Last batch index (exclusive) to submit.")
    # Misc
    p.add_argument("--seed", type=int, default=None, help="Shuffle dataset with this seed before batching.")
    return p.parse_args()


def main():
    args = parse_args()

    # Derive output prefix if not specified
    if args.s3_output_prefix is None:
        args.s3_output_prefix = args.s3_prefix.replace("input", "output")
        if args.s3_output_prefix == args.s3_prefix:
            args.s3_output_prefix = args.s3_prefix + "_output"

    # ---- Load dataset ------------------------------------------------
    logger.info("Loading dataset from: %s", args.dataset)
    dataset = Dataset.load_from_disk(args.dataset)
    logger.info("Dataset loaded: %d samples, columns: %s", len(dataset), dataset.column_names)

    if args.seed is not None:
        dataset = dataset.shuffle(seed=args.seed)
        logger.info("Dataset shuffled with seed=%d", args.seed)

    # ---- Create and upload batches -----------------------------------
    
    uploaded = create_and_upload_batches(
        dataset=dataset,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        batch_size=args.batch_size,
        start_index=args.start_index,
        max_samples=args.max_samples,
        model_id=args.model_id
    )
      
    # ---- Optionally submit jobs --------------------------------------
    if args.submit_jobs:
        num_batches = len(uploaded)
        batch_start = args.batch_start
        batch_end = args.batch_end if args.batch_end is not None else num_batches

        logger.info("Submitting batch jobs %d–%d ...", batch_start, batch_end - 1)
        submit_batch_jobs(
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_output_prefix=args.s3_output_prefix,
            num_batches=num_batches,
            role_arn=args.role_arn,
            model_id=args.model_id,
            job_name_prefix=args.job_name_prefix,
            batch_start=batch_start,
            batch_end=batch_end,
        )
        logger.info("All jobs submitted.")
    else:
        logger.info("Batches uploaded. Use --submit-jobs to submit inference jobs.")


if __name__ == "__main__":
    main()
