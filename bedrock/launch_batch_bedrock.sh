#!/bin/bash
# ===========================================================================
# run_all_batch_inference.sh
#
# Launches multiple Bedrock batch inference jobs for each dataset stored in
# s3://your-s3-bucket/vcape-dataset/. Datasets are downloaded locally, then
# uploaded as JSONL batches to s3://your-s3-bucket/2026/ and optionally
# submitted as Bedrock batch inference jobs.
#
# Prerequisites:
#   - AWS credentials configured (e.g., AWS_PROFILE=auditing_beta)
#   - Python environment with: boto3, datasets, Pillow, tqdm
#
# Usage:
#   # Dry run — only upload batches to S3, don't submit jobs
#   bash run_all_batch_inference.sh
#
#   # Upload and submit jobs
#   bash run_all_batch_inference.sh --submit
#
#   # Override defaults
#   S3_BUCKET=my-bucket BATCH_SIZE=200 bash run_all_batch_inference.sh --submit

#Optimized_QWEN is llama
# ===========================================================================

set -euo pipefail
# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
S3_BUCKET="${S3_BUCKET:-your-s3-bucket}"
S3_DATASET_PREFIX="${S3_DATASET_PREFIX:-vcape-dataset}"
S3_INPUT_BASE="${S3_INPUT_BASE:-autoqa-project/bedrock_input_datasets_optimized_Nova}"
S3_OUTPUT_BASE="${S3_OUTPUT_BASE:-autoqa-project/bedrock_output_datasets_prompt_optimized}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
ROLE_ARN="${ROLE_ARN:-arn:aws:iam::123456789012:role/YourBedrockBatchRole}"
MODEL_NAME="${MODEL_NAME:-us.amazon.nova-2-lite-}"
MODEL_ID="${MODEL_ID:-arn:aws:bedrock:us-east-1:123456789012:inference-profile}/${MODEL_NAME}${VERSION:-v1:0}"
SEED="${SEED:-42}"

# Path to the Python script (relative to this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_bedrock_batch_inference.py"

# Local directory to cache datasets downloaded from S3
LOCAL_DATASET_CACHE="${LOCAL_DATASET_CACHE:-$HOME/tmp/vcape-dataset-cache}"
FULL_DATASET_DIR="${FULL_DATASET_DIR:-${LOCAL_DATASET_CACHE}}"


# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SUBMIT_FLAG=""
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT_FLAG="--submit-jobs"
    echo ">>> Mode: UPLOAD + SUBMIT JOBS"
else
    echo ">>> Mode: UPLOAD ONLY (pass --submit to also submit jobs)"
fi

# ---------------------------------------------------------------------------
# Datasets to process
# Each entry: <dataset_subdir>  <job_name_prefix>
# ---------------------------------------------------------------------------
DATASETS=(
#    "vcape-r-20k    vcape-r-20k"
    "vcape-s-20k    vcape-s-20k"
)

# ---------------------------------------------------------------------------
# Run batch inference for each dataset
# ---------------------------------------------------------------------------
echo "========================================================"
echo "  Bedrock Batch Inference — Full Dataset Runner"
echo "========================================================"
echo "  S3 Bucket:    ${S3_BUCKET}"
echo "  S3 Datasets:  s3://${S3_BUCKET}/${S3_DATASET_PREFIX}/"
echo "  S3 Output:    s3://${S3_BUCKET}/${S3_OUTPUT_BASE}/"
echo "  Batch Size:   ${BATCH_SIZE}"
echo "  Model ID:     ${MODEL_ID}"
echo "  Seed:         ${SEED}"
echo "  Local Cache:  ${FULL_DATASET_DIR}"
echo "========================================================"
echo ""


# ---------------------------------------------------------------------------
# Run batch inference for each dataset
# ---------------------------------------------------------------------------
FAILED=0
TOTAL=0

for entry in "${DATASETS[@]}"; do
    # Parse entry
    read -r DATASET_SUBDIR JOB_PREFIX <<< "${entry}"

    DATASET_PATH="${FULL_DATASET_DIR}/${DATASET_SUBDIR}"
    S3_PREFIX="${S3_INPUT_BASE}/${DATASET_SUBDIR}"
    S3_OUTPUT_PREFIX="${S3_OUTPUT_BASE}/${DATASET_SUBDIR}/${MODEL_NAME}"

    TOTAL=$((TOTAL + 1))

    echo "--------------------------------------------------------"
    echo "  Dataset:      ${DATASET_PATH}"
    echo "  S3 Input:     s3://${S3_BUCKET}/${S3_PREFIX}/"
    echo "  S3 Output:    s3://${S3_BUCKET}/${S3_OUTPUT_PREFIX}/"
    echo "  Job Prefix:   ${JOB_PREFIX}"
    echo "--------------------------------------------------------"

    # Check dataset exists locally (should have been downloaded above)
    if [[ ! -d "${DATASET_PATH}" ]]; then
        echo "  ⚠️  Dataset directory not found: ${DATASET_PATH} — SKIPPING"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Build the command
    CMD=(
        python "${PYTHON_SCRIPT}"
        --dataset "${DATASET_PATH}"
        --s3-bucket "${S3_BUCKET}"
        --s3-prefix "${S3_PREFIX}"
        --s3-output-prefix "${S3_OUTPUT_PREFIX}"
        --batch-size "${BATCH_SIZE}"
        --seed "${SEED}"
        --role-arn "${ROLE_ARN}"
        --model-id "${MODEL_ID}"
        --job-name-prefix "${JOB_PREFIX}"
    )

    # Add submit flag if requested
    if [[ -n "${SUBMIT_FLAG}" ]]; then
        CMD+=("${SUBMIT_FLAG}")
    fi

    echo "  Running: ${CMD[*]}"
    echo ""

    # Execute
    if "${CMD[@]}"; then
        echo ""
        echo "  ✅ ${DATASET_SUBDIR} completed successfully."
    else
        echo ""
        echo "  ❌ ${DATASET_SUBDIR} FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "========================================================"
echo "  Summary: ${TOTAL} datasets processed, ${FAILED} failed"
echo "========================================================"

if [[ ${FAILED} -gt 0 ]]; then
    exit 1
fi
