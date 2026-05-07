#!/usr/bin/env python3
"""
Unified Multi-Threaded Dataset Evaluator

Reads HuggingFace-format datasets and evaluates them using either:
  - bedrock: AWS Bedrock Claude (BedrockImageEvaluator)
  - autoqa:  Local vLLM server  (autoqa_evaluator style)

Uses ThreadPoolExecutor for concurrent API calls, with progress tracking,
automatic retries, and result saving.

Usage:
    # Bedrock evaluator
    python evaluate_dataset.py --evaluator bedrock --dataset ./dataset \
        --model us.anthropic.claude-sonnet-4-5-20250929-v1:0 --num-threads 12

    # AutoQA / vLLM evaluator
    python evaluate_dataset.py --evaluator autoqa --dataset ./dataset \
        --model Qwen/Qwen3-VL-8B-Instruct --vllm-port 8765 --num-threads 40
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from datasets import Dataset
from PIL import Image
from rich import print_json
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluate_dataset.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import Bedrock evaluator (only needed when --evaluator bedrock)
# ---------------------------------------------------------------------------
try:
    from bedrock_evaluator import BedrockImageEvaluator, OutputLabel
except ImportError:
    BedrockImageEvaluator = None
    OutputLabel = None


# ===================================================================
#  AUTOQA  (vLLM)  helpers
# ===================================================================

def _pil_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _half_resolution(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
    return img


def _make_autoqa_conversation(sample: Dict) -> list:
    """Build the chat messages for vLLM (OpenAI-compatible API)."""
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

    imgs = {}
    for key in ("source_image_main", "source_image_additional", "generated_image"):
        if key in sample and sample[key] is not None:
            imgs[key] = _half_resolution(sample[key])

    content_parts = []
    for label, key in [
        ("SOURCE IMAGE", "source_image_main"),
        ("ADDITIONAL SOURCE IMAGE", "source_image_additional"),
        ("GENERATED IMAGE", "generated_image"),
    ]:
        if key in imgs:
            content_parts.append({"type": "text", "text": f"\n**{label}**"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_pil_to_base64(imgs[key])}"},
            })
    content_parts.append({"type": "text", "text": PROMPT})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]


def _extract_autoqa_answer(output: str) -> str:
    match = re.search(r"</reasoning>\s*(.*)", output, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().lower().replace("'", "")
    return "no"


def _call_vllm_api(
    messages: list,
    port: int = 8000,
    model_name: str = "",
    temperature: float = 0.8,
    max_tokens: int = 1024,
    top_p: float = 0.8,
) -> str:
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "presence_penalty": 1.5,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ===================================================================
#  BEDROCK helpers
# ===================================================================

def _extract_bedrock_answer(result: dict) -> Tuple[str, str]:
    """Return (label, reasoning) from Bedrock JSON response."""
    if OutputLabel is None:
        return "", ""
    valid = [item.label for item in OutputLabel]
    if "recommendation" not in result or result["recommendation"] not in valid:
        logger.warning("Bedrock output label not recognized: %s", result.get("recommendation"))
        return "", ""
    reasoning = result.get("verdict", "")
    return result["recommendation"], reasoning


# ===================================================================
#  Generic single-sample processors
# ===================================================================

def process_sample_autoqa(
    sample: Dict, model_name: str, port: int, verbose: bool = False
) -> Dict:
    messages = _make_autoqa_conversation(sample)
    raw = _call_vllm_api(messages, port=port, model_name=model_name)
    answer = _extract_autoqa_answer(raw)
    if verbose:
        gt = sample.get("is_generation_successful", sample.get("label", "?"))
        logger.info("AutoQA response: %s | GT: %s", answer, gt)
    return {
        **sample,
        "model_response": raw,
        "model_prediction_is_generation_successfull": answer.strip() == "yes",
        "model_predicted_label": answer.strip(),
    }


def process_sample_bedrock(
    sample: Dict, evaluator: "BedrockImageEvaluator", verbose: bool = False
) -> Dict:
    # Downscale source images (generated_image kept full res like original code)
    for key in ("source_image_main", "source_image_additional"):
        if key in sample and sample[key] is not None:
            sample[key] = _half_resolution(sample[key])

    result = evaluator.process_single_sample(sample_data=sample)
    label, reasoning = _extract_bedrock_answer(result)
    if verbose:
        gt = sample.get("is_generation_successful", sample.get("label", "?"))
        logger.info("Bedrock label: %s | GT: %s", label, gt)
    is_acceptable = (label == OutputLabel.ACCEPTABLE.label) if OutputLabel else False
    return {
        **sample,
        "model_response": reasoning,
        "model_predicted_label": label,
        "model_prediction_is_generation_successfull": is_acceptable,
    }


# ===================================================================
#  Threaded runner
# ===================================================================

def run_threaded_evaluation(
    dataset: Dataset,
    evaluator_type: str,
    model_name: str,
    num_threads: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
    # autoqa-specific
    vllm_port: int = 8765,
    # bedrock-specific
    bedrock_region: str = "us-east-1",
) -> List[Dict]:
    """
    Evaluate every sample in *dataset* using a ThreadPoolExecutor.

    Returns a list of result dicts (one per sample, in order).
    """
    total = len(dataset)
    logger.info("Starting threaded evaluation: %d samples, %d threads, evaluator=%s",
                total, num_threads, evaluator_type)

    # Pre-create evaluator for bedrock (thread-safe because boto3 client
    # is created lazily per call via the property).
    bedrock_eval = None
    if evaluator_type == "bedrock":
        if BedrockImageEvaluator is None:
            raise ImportError(
                "bedrock_evaluator module not found. Make sure bedrock_evaluator.py "
                "is in the same directory or on PYTHONPATH."
            )
        bedrock_eval = BedrockImageEvaluator(
            region_name=bedrock_region, model_name=model_name
        )

    # ---- worker function with retries --------------------------------
    def _worker(index: int) -> Tuple[int, Dict]:
        sample = dataset[index]
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                if evaluator_type == "bedrock":
                    return index, process_sample_bedrock(sample, bedrock_eval, verbose)
                else:
                    return index, process_sample_autoqa(sample, model_name, vllm_port, verbose)
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Sample %d attempt %d/%d failed: %s", index, attempt, max_retries, exc
                )
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
        # All retries exhausted
        logger.error("Sample %d FAILED after %d retries: %s", index, max_retries, last_err)
        return index, {
            **sample,
            "model_response": f"ERROR: {last_err}",
            "model_predicted_label": "ERROR",
            "model_prediction_is_generation_successfull": False,
        }

    # ---- dispatch work -----------------------------------------------
    results: List[Optional[Dict]] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = {pool.submit(_worker, i): i for i in range(total)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                logger.info("Progress: %d / %d  (%.1f%%)", completed, total, 100 * completed / total)

    return results


# ===================================================================
#  Ground-truth helper
# ===================================================================

def _get_ground_truth_bool(sample: Dict) -> bool:
    """
    Extract ground truth as a boolean from a sample dict.
    Supports both column formats:
      - 'is_generation_successful': bool
      - 'label': str ('accept' -> True, 'reject' -> False)
    """
    if "is_generation_successful" in sample:
        return bool(sample["is_generation_successful"])
    if "label" in sample:
        return sample["label"].lower().strip() == "accept"
    raise KeyError(
        f"Cannot find ground truth column. Available keys: {list(sample.keys())}. "
        "Expected 'is_generation_successful' or 'label'."
    )


# ===================================================================
#  Metrics
# ===================================================================

def calculate_metrics(results: List[Dict], evaluator_type: str) -> Dict:
    ground_truth = [_get_ground_truth_bool(r) for r in results]

    if evaluator_type == "bedrock" and OutputLabel is not None:
        predictions = [
            r.get("model_predicted_label") == OutputLabel.ACCEPTABLE.label
            for r in results
        ]
    else:
        predictions = [bool(r.get("model_prediction_is_generation_successfull")) for r in results]

    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(predictions),
        "positive_samples": int(sum(ground_truth)),
        "negative_samples": int(len(ground_truth) - sum(ground_truth)),
    }


# ===================================================================
#  Save results
# ===================================================================

def save_results(
    results: List[Dict],
    metrics: Dict,
    model_name: str,
    evaluator_type: str,
    output_base_dir: str = "evaluation_results",
    dataset_name: str = "",
) -> str:
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{safe_model}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # metrics.json
    meta = {
        "model_name": model_name,
        "evaluator_type": evaluator_type,
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "evaluation_date": datetime.now().isoformat(),
        "metrics": metrics,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # summary.txt
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluator: {evaluator_type}\n")
        f.write(f"Evaluation Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"\n{'='*60}\nMETRICS\n{'='*60}\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}\n")
        f.write(f"  FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}\n")
    logger.info("Summary saved to %s", summary_path)

    # Per-sample reasoning
    reasoning_dir = os.path.join(output_dir, "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)
    for ix, r in enumerate(results):
        with open(os.path.join(reasoning_dir, f"{ix:09d}.txt"), "w") as f:
            f.write(str(r.get("model_response", "")))

    # Per-sample predictions as JSONL (excludes images)
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, "w") as f:
        for ix, r in enumerate(results):
            row = {
                "index": ix,
                "ground_truth": _get_ground_truth_bool(r),
                "predicted_label": r.get("model_predicted_label", ""),
                "predicted_success": r.get("model_prediction_is_generation_successfull"),
                "reasoning": str(r.get("model_response", ""))[:2000],
            }
            f.write(json.dumps(row) + "\n")
    logger.info("Per-sample predictions saved to %s", predictions_path)

    return output_dir


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-threaded HuggingFace dataset evaluator (Bedrock or AutoQA/vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--evaluator", type=str, required=True, choices=["bedrock", "autoqa"],
        help="Which evaluator backend to use.",
    )
    p.add_argument(
        "--dataset", type=str, required=True,
        help="Path to a HuggingFace Dataset on disk (load_from_disk format).",
    )
    p.add_argument(
        "--model", type=str, required=True,
        help="Model identifier. For bedrock: model ARN suffix. For autoqa: model path served by vLLM.",
    )
    p.add_argument("--output-dir", type=str, default="evaluation_results", help="Base output directory.")
    p.add_argument("--num-threads", type=int, default=12, help="Number of concurrent threads.")
    p.add_argument("--max-retries", type=int, default=3, help="Max retries per sample on failure.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffle.")
    p.add_argument("--debug", action="store_true", help="Run on a small subset (4 samples).")
    p.add_argument("--verbose", action="store_true", help="Print per-sample predictions.")
    # AutoQA / vLLM specific
    p.add_argument("--vllm-port", type=int, default=8765, help="vLLM server port (autoqa only).")
    # Bedrock specific
    p.add_argument("--bedrock-region", type=str, default="us-east-1", help="AWS region (bedrock only).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load dataset ------------------------------------------------
    logger.info("Loading dataset from: %s", args.dataset)
    dataset = Dataset.load_from_disk(args.dataset)
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(3000))
    if args.debug:
        dataset = dataset.select(range(min(4, len(dataset))))
        logger.info("DEBUG mode: using %d samples", len(dataset))

    logger.info("Dataset loaded: %d samples, columns: %s", len(dataset), dataset.column_names)

    # ---- Run evaluation ----------------------------------------------
    logger.info("=" * 60)
    logger.info("Evaluator : %s", args.evaluator)
    logger.info("Model     : %s", args.model)
    logger.info("Threads   : %d", args.num_threads)
    logger.info("Retries   : %d", args.max_retries)
    logger.info("=" * 60)

    t0 = time.time()
    results = run_threaded_evaluation(
        dataset=dataset,
        evaluator_type=args.evaluator,
        model_name=args.model,
        num_threads=args.num_threads,
        max_retries=args.max_retries,
        verbose=args.verbose or args.debug,
        vllm_port=args.vllm_port,
        bedrock_region=args.bedrock_region,
    )
    elapsed = time.time() - t0
    logger.info("Evaluation completed in %.1f seconds (%.2f samples/sec)", elapsed, len(results) / elapsed)

    # ---- Metrics -----------------------------------------------------
    logger.info("Calculating metrics ...")
    metrics = calculate_metrics(results, args.evaluator)
    print_json(json.dumps(metrics))

    # ---- Save --------------------------------------------------------
    out = save_results(
        results=results,
        metrics=metrics,
        model_name=args.model,
        evaluator_type=args.evaluator,
        output_base_dir=args.output_dir,
        dataset_name=args.dataset,
    )
    logger.info("✓ Evaluation complete!  Results saved to: %s", out)


if __name__ == "__main__":
    main()
