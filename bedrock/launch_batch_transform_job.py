#!/usr/bin/env python3
"""
Launch a SageMaker Batch Transform Job

Creates and launches a SageMaker Batch Transform job that reads JSONL
payloads from S3, runs inference through a SageMaker model, and writes
results back to S3.

Usage:
    # Basic usage
    python launch_batch_transform_job.py \
        --model-name Qwen-Qwen3-VL-8B-Instruct-FP8 \
        --input-path s3://your-s3-bucket/input_datasets/ \
        --output-path s3://your-s3-bucket/output_datasets/

    # With custom instance configuration
    python launch_batch_transform_job.py \
        --model-name Qwen-Qwen3-VL-4B-Instruct-FP8 \
        --input-path s3://bucket/input/ \
        --output-path s3://bucket/output/ \
        --instance-type ml.g5.12xlarge \
        --instance-count 5 \
        --max-concurrent-transforms 2 \
        --job-name-prefix Qwen3-VL-4B-eval \
        --region us-east-1
"""

import argparse
import logging
from datetime import datetime

import boto3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("launch_batch_transform.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
DEFAULT_INSTANCE_TYPE = "ml.g5.12xlarge"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_MAX_CONCURRENT_TRANSFORMS = 1
DEFAULT_JOB_NAME_PREFIX = "Qwen3-VL-batch"
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_PAYLOAD_MB = 10


# ===================================================================
#  Batch Transform Job
# ===================================================================

def launch_batch_transform_job(
    model_name: str,
    s3_input_path: str,
    s3_output_path: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    instance_count: int = DEFAULT_INSTANCE_COUNT,
    max_concurrent_transforms: int = DEFAULT_MAX_CONCURRENT_TRANSFORMS,
    job_name_prefix: str = DEFAULT_JOB_NAME_PREFIX,
    max_payload_mb: int = DEFAULT_MAX_PAYLOAD_MB,
    region: str = DEFAULT_REGION,
) -> str:
    """
    Launch a SageMaker Batch Transform job.

    The job reads JSONL from *s3_input_path*, passes each line through the
    SageMaker model, and writes JSONL results to *s3_output_path*.

    Args:
        model_name:               Name of the SageMaker model to use.
        s3_input_path:            S3 URI for input data (e.g. ``s3://bucket/input/``).
        s3_output_path:           S3 URI for output data (e.g. ``s3://bucket/output/``).
        instance_type:            EC2 instance type for the transform job.
        instance_count:           Number of instances.
        max_concurrent_transforms: Max concurrent transforms per instance.
        job_name_prefix:          Prefix for the transform job name.
        max_payload_mb:           Max payload size in MB.
        region:                   AWS region.

    Returns:
        The transform job name.
    """
    sagemaker_client = boto3.client("sagemaker", region_name=region)

    # Generate unique job name with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    transform_job_name = f"{job_name_prefix}-{timestamp}"

    logger.info("=" * 60)
    logger.info("Launching Batch Transform Job")
    logger.info("=" * 60)
    logger.info("  Job Name  : %s", transform_job_name)
    logger.info("  Model     : %s", model_name)
    logger.info("  Input     : %s", s3_input_path)
    logger.info("  Output    : %s", s3_output_path)
    logger.info("  Instance  : %s (count: %d)", instance_type, instance_count)
    logger.info("  Concurrency: %d", max_concurrent_transforms)
    logger.info("  Region    : %s", region)

    try:
        response = sagemaker_client.create_transform_job(
            TransformJobName=transform_job_name.replace("-",""),
            ModelName=model_name,
            TransformInput={
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": s3_input_path,
                        "S3DataType": "S3Prefix",
                    }
                },
                "ContentType": "application/jsonlines",
                "SplitType": "Line",
            },
            TransformOutput={
                "S3OutputPath": s3_output_path,
                "AssembleWith": "Line",
                "Accept": "application/jsonlines",
            },
            BatchStrategy="SingleRecord",
            MaxPayloadInMB=max_payload_mb,
            MaxConcurrentTransforms=max_concurrent_transforms,
            TransformResources={
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
            },
            DataProcessing={
                "InputFilter": "$.payload",
                "OutputFilter": "$",
                "JoinSource": "Input",
            },
        )

        job_arn = response["TransformJobArn"]
        logger.info("✓ Batch Transform Job created successfully!")
        logger.info("  Job ARN: %s", job_arn)
        logger.info(
            "  Monitor with: aws sagemaker describe-transform-job "
            "--transform-job-name %s",
            transform_job_name,
        )
        return transform_job_name

    except Exception as exc:
        logger.error("✗ Error creating Batch Transform Job: %s", exc)
        raise


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Launch a SageMaker Batch Transform Job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-name", type=str, required=True,
        help="Name of the SageMaker model to use.",
    )
    p.add_argument(
        "--input-path", type=str, required=True,
        help="S3 URI for input data (e.g. s3://bucket/input/).",
    )
    p.add_argument(
        "--output-path", type=str, required=True,
        help="S3 URI for output data (e.g. s3://bucket/output/).",
    )
    p.add_argument(
        "--instance-type", type=str, default=DEFAULT_INSTANCE_TYPE,
        help=f"Instance type (default: {DEFAULT_INSTANCE_TYPE}).",
    )
    p.add_argument(
        "--instance-count", type=int, default=DEFAULT_INSTANCE_COUNT,
        help=f"Number of instances (default: {DEFAULT_INSTANCE_COUNT}).",
    )
    p.add_argument(
        "--max-concurrent-transforms", type=int, default=DEFAULT_MAX_CONCURRENT_TRANSFORMS,
        help=f"Max concurrent transforms per instance (default: {DEFAULT_MAX_CONCURRENT_TRANSFORMS}).",
    )
    p.add_argument(
        "--job-name-prefix", type=str, default=DEFAULT_JOB_NAME_PREFIX,
        help=f"Prefix for the transform job name (default: {DEFAULT_JOB_NAME_PREFIX}).",
    )
    p.add_argument(
        "--max-payload-mb", type=int, default=DEFAULT_MAX_PAYLOAD_MB,
        help=f"Max payload size in MB (default: {DEFAULT_MAX_PAYLOAD_MB}).",
    )
    p.add_argument(
        "--region", type=str, default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION}).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    job_name = launch_batch_transform_job(
        model_name=args.model_name,
        s3_input_path=args.input_path,
        s3_output_path=args.output_path,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        max_concurrent_transforms=args.max_concurrent_transforms,
        job_name_prefix=args.job_name_prefix,
        max_payload_mb=args.max_payload_mb,
        region=args.region,
    )
    logger.info("Transform job launched: %s", job_name)


if __name__ == "__main__":
    main()
