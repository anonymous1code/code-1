#!/bin/bash
# ===========================================================================
# run_batch_pipeline.sh
#
# End-to-end SageMaker Batch Transform pipeline for Qwen VL models.
# Given a HuggingFace model name, this script:
#   1. Creates a SageMaker model (DJL/LMI + vLLM container)
#   2. Prepares batch transform payloads from two HuggingFace datasets
#      stored in s3://your-s3-bucket/vcape-dataset/
#   3. Launches batch transform jobs, outputting to s3://your-s3-bucket/2026/
#
# Prerequisites:
#   - AWS credentials configured (e.g., aws configure ...)
#   - Python environment with: boto3, datasets, Pillow, tqdm
#
# Usage:
#   # Full pipeline (create model + prepare payloads + launch jobs)
#   bash run_batch_pipeline.sh Qwen/Qwen3-VL-8B-Instruct-FP8
#
#   # Prepare payloads only (skip model creation and job launch)
#   bash run_batch_pipeline.sh Qwen/Qwen3-VL-8B-Instruct-FP8 --prepare-only
#
#   # Skip model creation (model already exists)
#   bash run_batch_pipeline.sh Qwen/Qwen3-VL-8B-Instruct-FP8 --skip-model-creation
#
#   # Override defaults via environment variables
#   INSTANCE_TYPE=ml.g5.48xlarge INSTANCE_COUNT=3 \
#       bash run_batch_pipeline.sh Qwen/Qwen3-VL-32B-Instruct
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate input
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 <hf-model-id> [--prepare-only|--skip-model-creation]"
    echo ""
    echo "  <hf-model-id>          HuggingFace model identifier"
    echo "                         e.g. Qwen/Qwen3-VL-8B-Instruct-FP8"
    echo ""
    echo "Options:"
    echo "  --prepare-only         Only prepare & upload payloads (no model creation, no job launch)"
    echo "  --skip-model-creation  Skip SageMaker model creation (model already exists)"
    echo ""
    echo "Environment overrides:"
    echo "  S3_BUCKET, EXECUTION_ROLE_ARN, INSTANCE_TYPE, INSTANCE_COUNT,"
    echo "  MAX_CONCURRENT_TRANSFORMS, BATCH_SIZE, SEED, REGION"
    exit 1
fi

HF_MODEL_ID="$1"
shift

# Parse optional flags
PREPARE_ONLY=false
SKIP_MODEL_CREATION=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prepare-only)
            PREPARE_ONLY=true
            shift
            ;;
        --skip-model-creation)
            SKIP_MODEL_CREATION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Derive SageMaker model name from HF model ID
# e.g. "Qwen/Qwen3-VL-8B-Instruct-FP8" → "Qwen-Qwen3-VL-8B-Instruct-FP8"
# ---------------------------------------------------------------------------
SAGEMAKER_MODEL_NAME="${HF_MODEL_ID//\//-}"

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
S3_BUCKET="${S3_BUCKET:-your-s3-bucket}"
S3_DATASET_PREFIX="${S3_DATASET_PREFIX:-vcape-dataset}"
S3_INPUT_BASE="${S3_INPUT_BASE:-2026/vllm_input_datasets/${SAGEMAKER_MODEL_NAME}}"
S3_OUTPUT_BASE="${S3_OUTPUT_BASE:-2026/vllm_output_datasets/${SAGEMAKER_MODEL_NAME}}"
EXECUTION_ROLE_ARN="${EXECUTION_ROLE_ARN:-arn:aws:iam::123456789012:role/YourSageMakerRole}"
REGION="${REGION:-us-east-1}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
SEED="${SEED:-42}"

# SageMaker Batch Transform settings
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g5.12xlarge}"
INSTANCE_COUNT="${INSTANCE_COUNT:-1}"
MAX_CONCURRENT_TRANSFORMS="${MAX_CONCURRENT_TRANSFORMS:-1}"

# Local dataset cache
LOCAL_DATASET_CACHE="${LOCAL_DATASET_CACHE:-/tmp/vcape-dataset-cache}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Datasets to process
# ---------------------------------------------------------------------------
DATASETS=(
    
    "vcape-s-20k"
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "========================================================"
echo "  SageMaker Batch Transform Pipeline (vLLM)"
echo "========================================================"
echo "  HF Model ID     : ${HF_MODEL_ID}"
echo "  SageMaker Model  : ${SAGEMAKER_MODEL_NAME}"
echo "  S3 Bucket        : ${S3_BUCKET}"
echo "  S3 Datasets      : s3://${S3_BUCKET}/${S3_DATASET_PREFIX}/"
echo "  S3 Input Prefix  : s3://${S3_BUCKET}/${S3_INPUT_BASE}/"
echo "  S3 Output Prefix : s3://${S3_BUCKET}/${S3_OUTPUT_BASE}/"
echo "  Instance         : ${INSTANCE_TYPE} x${INSTANCE_COUNT}"
echo "  Concurrency      : ${MAX_CONCURRENT_TRANSFORMS}"
echo "  Batch Size       : ${BATCH_SIZE}"
echo "  Seed             : ${SEED}"
echo "  Region           : ${REGION}"
echo "  Prepare Only     : ${PREPARE_ONLY}"
echo "  Skip Model       : ${SKIP_MODEL_CREATION}"
echo "========================================================"
echo ""
# ===========================================================================
# Step 1: Create SageMaker Model
# ===========================================================================
if [[ "${PREPARE_ONLY}" == "false" && "${SKIP_MODEL_CREATION}" == "false" ]]; then
    echo ">>> Step 1: Creating SageMaker model '${SAGEMAKER_MODEL_NAME}' ..."
    python "${SCRIPT_DIR}/create_qwen_sagemaker_model.py" \
        --model-name "${SAGEMAKER_MODEL_NAME}" \
        --hf-model-id "${HF_MODEL_ID}" \
        --execution-role-arn "${EXECUTION_ROLE_ARN}" \
        --region "${REGION}"
    echo "  ✅ SageMaker model created."
    echo ""
else
    echo ">>> Step 1: Skipping SageMaker model creation."
    echo ""
fi
# ===========================================================================
# Step 2: Download datasets from S3
# ===========================================================================
echo ">>> Step 2: Downloading datasets from s3://${S3_BUCKET}/${S3_DATASET_PREFIX}/ ..."
mkdir -p "${LOCAL_DATASET_CACHE}"

for DATASET_NAME in "${DATASETS[@]}"; do
    S3_DATASET_URI="s3://${S3_BUCKET}/${S3_DATASET_PREFIX}/${DATASET_NAME}/"
    LOCAL_DEST="${LOCAL_DATASET_CACHE}/${DATASET_NAME}"

    if [[ -d "${LOCAL_DEST}" ]]; then
        echo "  ✔ ${DATASET_NAME} already cached at ${LOCAL_DEST} — skipping download"
    else
        echo "  ⬇ Syncing ${S3_DATASET_URI} → ${LOCAL_DEST} ..."
        aws s3 sync "${S3_DATASET_URI}" "${LOCAL_DEST}/"
        echo "  ✔ ${DATASET_NAME} downloaded."
    fi
done
echo ""
# ===========================================================================
# Step 3: Prepare batch transform payloads & upload to S3
# ===========================================================================
echo ">>> Step 3: Preparing batch transform payloads ..."
echo ""

FAILED=0
TOTAL=0

# for DATASET_NAME in "${DATASETS[@]}"; do
#     DATASET_PATH="${LOCAL_DATASET_CACHE}/${DATASET_NAME}"
#     S3_PREFIX="${S3_INPUT_BASE}/${DATASET_NAME}"

#     TOTAL=$((TOTAL + 1))

#     echo "--------------------------------------------------------"
#     echo "  Dataset   : ${DATASET_NAME}"
#     echo "  Local Path: ${DATASET_PATH}"
#     echo "  S3 Input  : s3://${S3_BUCKET}/${S3_PREFIX}/"
#     echo "--------------------------------------------------------"

#     if [[ ! -d "${DATASET_PATH}" ]]; then
#         echo "  ⚠️  Dataset directory not found: ${DATASET_PATH} — SKIPPING"
#         FAILED=$((FAILED + 1))
#         continue
#     fi

#     if python "${SCRIPT_DIR}/prepare_inference_batch_transform_payload.py" \
#         --dataset "${DATASET_PATH}" \
#         --s3-bucket "${S3_BUCKET}" \
#         --s3-prefix "${S3_PREFIX}" \
#         --batch-size "${BATCH_SIZE}" \
#         --seed "${SEED}"; then
#         echo "  ✅ ${DATASET_NAME} payloads uploaded."
#     else
#         echo "  ❌ ${DATASET_NAME} payload preparation FAILED."
#         FAILED=$((FAILED + 1))
#     fi
#     echo ""
# done

# ===========================================================================
# Step 4: Launch batch transform jobs
# ===========================================================================
if [[ "${PREPARE_ONLY}" == "true" ]]; then
    echo ">>> Step 4: Skipping job launch (--prepare-only mode)."
    echo ""
else
    echo ">>> Step 4: Launching batch transform jobs ..."
    echo ""

    for DATASET_NAME in "${DATASETS[@]}"; do
        S3_INPUT="s3://${S3_BUCKET}/${S3_INPUT_BASE}/${DATASET_NAME}/"
        S3_OUTPUT="s3://${S3_BUCKET}/${S3_OUTPUT_BASE}/${DATASET_NAME}/"
        JOB_PREFIX="${SAGEMAKER_MODEL_NAME}-${DATASET_NAME}"

        echo "--------------------------------------------------------"
        echo "  Dataset    : ${DATASET_NAME}"
        echo "  S3 Input   : ${S3_INPUT}"
        echo "  S3 Output  : ${S3_OUTPUT}"
        echo "  Job Prefix : ${JOB_PREFIX}"
        echo "--------------------------------------------------------"

        if python "${SCRIPT_DIR}/launch_batch_transform_job.py" \
            --model-name "${SAGEMAKER_MODEL_NAME}" \
            --input-path "${S3_INPUT}" \
            --output-path "${S3_OUTPUT}" \
            --instance-type "${INSTANCE_TYPE}" \
            --instance-count "${INSTANCE_COUNT}" \
            --max-concurrent-transforms "${MAX_CONCURRENT_TRANSFORMS}" \
            --job-name-prefix "${JOB_PREFIX}" \
            --region "${REGION}"; then
            echo "  ✅ ${DATASET_NAME} batch transform job launched."
        else
            echo "  ❌ ${DATASET_NAME} job launch FAILED."
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
fi

# ===========================================================================
# Summary
# ===========================================================================
echo "========================================================"
echo "  Pipeline Summary"
echo "========================================================"
echo "  Model           : ${HF_MODEL_ID}"
echo "  Datasets        : ${DATASETS[*]}"
echo "  Total steps     : ${TOTAL}"
echo "  Failures        : ${FAILED}"
if [[ "${PREPARE_ONLY}" == "false" ]]; then
    echo "  Output location : s3://${S3_BUCKET}/${S3_OUTPUT_BASE}/"
    echo ""
    echo "  Monitor jobs with:"
    echo "    aws sagemaker list-transform-jobs --name-contains ${SAGEMAKER_MODEL_NAME} --region ${REGION}"
fi
echo "========================================================"

if [[ ${FAILED} -gt 0 ]]; then
    echo "⚠️  ${FAILED} step(s) failed. Check logs above."
    exit 1
else
    echo "✅ All steps completed successfully!"
fi
