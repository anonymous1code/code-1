#!/usr/bin/env python3
"""
Create a SageMaker Model for Qwen VL (vLLM / LMI container)

Registers a SageMaker model backed by a DJL/LMI inference container,
configured to serve a Qwen Vision-Language model via the vLLM backend.

Usage:
    # Create model with defaults
    python create_qwen_sagemaker_model.py \
        --model-name Qwen-Qwen3-VL-8B-Instruct-FP8 \
        --hf-model-id Qwen/Qwen3-VL-8B-Instruct-FP8 \
        --execution-role-arn arn:aws:iam::123456789012:role/YourSageMakerRole

    # Override container image and region
    python create_qwen_sagemaker_model.py \
        --model-name Qwen-Qwen3-VL-32B \
        --hf-model-id Qwen/Qwen3-VL-32B-Instruct \
        --execution-role-arn arn:aws:iam::123456789012:role/YourSageMakerRole \
        --ecr-image 763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi20.0.0-cu128-v1.0 \
        --region us-east-1
"""

import argparse
import json
import logging
import sys

import boto3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("create_sagemaker_model.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
DEFAULT_ECR_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "djl-inference:0.36.0-lmi20.0.0-cu128-v1.0"
)
DEFAULT_REGION = "us-east-1"


# ===================================================================
#  Model creation
# ===================================================================

def create_sagemaker_model(
    model_name: str,
    hf_model_id: str,
    execution_role_arn: str,
    ecr_image: str = DEFAULT_ECR_IMAGE,
    region: str = DEFAULT_REGION,
    extra_env: dict = None,
) -> str:
    """
    Register a SageMaker model that uses a DJL/LMI container to serve
    a HuggingFace Vision-Language model.

    Args:
        model_name:         SageMaker model name (must be unique in the account/region).
        hf_model_id:        HuggingFace model identifier (e.g. ``Qwen/Qwen3-VL-8B-Instruct-FP8``).
        execution_role_arn: IAM role ARN with SageMaker permissions.
        ecr_image:          ECR URI of the DJL/LMI inference container.
        region:             AWS region.
        extra_env:          Optional dict of additional environment variables for the container.

    Returns:
        The Model ARN string.
    """
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    environment = {"HF_MODEL_ID": hf_model_id}
    if extra_env:
        environment.update(extra_env)

    logger.info("Creating SageMaker model '%s' ...", model_name)
    logger.info("  HF Model ID : %s", hf_model_id)
    logger.info("  ECR Image   : %s", ecr_image)
    logger.info("  Role ARN    : %s", execution_role_arn)
    logger.info("  Region      : %s", region)
    if extra_env:
        logger.info("  Extra env   : %s", json.dumps(extra_env))

    response = sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": ecr_image,
            "Environment": environment,
        },
        ExecutionRoleArn=execution_role_arn,
    )

    model_arn = response["ModelArn"]
    logger.info("✓ Model created successfully: %s", model_arn)
    return model_arn


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Create a SageMaker Model for Qwen VL (vLLM / LMI container)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-name", type=str, required=True,
        help="SageMaker model name (must be unique in the account/region).",
    )
    p.add_argument(
        "--hf-model-id", type=str, required=True,
        help="HuggingFace model identifier (e.g. Qwen/Qwen3-VL-8B-Instruct-FP8).",
    )
    p.add_argument(
        "--execution-role-arn", type=str, required=True,
        help="IAM role ARN with SageMaker permissions.",
    )
    p.add_argument(
        "--ecr-image", type=str, default=DEFAULT_ECR_IMAGE,
        help=f"ECR URI of the DJL/LMI inference container. Default: {DEFAULT_ECR_IMAGE}",
    )
    p.add_argument(
        "--region", type=str, default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION}).",
    )
    # Optional LMI tuning knobs exposed as CLI flags
    p.add_argument(
        "--tensor-parallel-degree", type=str, default=None,
        help="OPTION_TENSOR_PARALLEL_DEGREE for the LMI container (e.g. 'max').",
    )
    p.add_argument(
        "--max-model-len", type=str, default=None,
        help="OPTION_MAX_MODEL_LEN — max total sequence length for KV cache.",
    )
    p.add_argument(
        "--rolling-batch", type=str, default=None,
        help="OPTION_ROLLING_BATCH backend (e.g. 'vllm').",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build optional env dict from CLI flags
    extra_env = {}
    if args.tensor_parallel_degree:
        extra_env["OPTION_TENSOR_PARALLEL_DEGREE"] = args.tensor_parallel_degree
    if args.max_model_len:
        extra_env["OPTION_MAX_MODEL_LEN"] = args.max_model_len
    if args.rolling_batch:
        extra_env["OPTION_ROLLING_BATCH"] = args.rolling_batch

    model_arn = create_sagemaker_model(
        model_name=args.model_name,
        hf_model_id=args.hf_model_id,
        execution_role_arn=args.execution_role_arn,
        ecr_image=args.ecr_image,
        region=args.region,
        extra_env=extra_env if extra_env else None,
    )
    print(f"Model ARN: {model_arn}")


if __name__ == "__main__":
    main()
