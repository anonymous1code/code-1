
#!/bin/bash

ROLE_ARN="arn:aws:iam::123456789012:role/YourGlueRole"
SCRIPT_S3="s3://your-s3-bucket/glue-scripts/extract_predictions.py"

# Upload the script to S3
aws s3 cp extract_predictions.py "$SCRIPT_S3"

# --- Job 1: bedrock_output_datasets (original prompts) ---
JOB_NAME_1="autoqa-batch-prediction-extractor"
S3_INPUT_1="s3://your-s3-bucket/autoqa-project/bedrock_output_datasets/"
S3_OUTPUT_1="s3://your-s3-bucket/autoqa-project/results/predictions/"

aws glue create-job \
  --name "$JOB_NAME_1" \
  --role "$ROLE_ARN" \
  --command '{
    "Name": "glueetl",
    "ScriptLocation": "'"$SCRIPT_S3"'",
    "PythonVersion": "3"
  }' \
  --glue-version "4.0" \
  --number-of-workers 10 \
  --worker-type "G.1X" \
  --default-arguments '{
    "--S3_INPUT_PREFIX": "'"$S3_INPUT_1"'",
    "--S3_OUTPUT_PATH": "'"$S3_OUTPUT_1"'",
    "--job-language": "python",
    "--enable-metrics": "true",
    "--enable-continuous-cloudwatch-log": "true"
  }' 2>/dev/null || echo "Job $JOB_NAME_1 already exists, skipping creation."

aws glue start-job-run \
  --job-name "$JOB_NAME_1" \
  --arguments '{
    "--S3_INPUT_PREFIX": "'"$S3_INPUT_1"'",
    "--S3_OUTPUT_PATH": "'"$S3_OUTPUT_1"'"
  }'

echo "Job 1 ($JOB_NAME_1) started."
echo "  Input:  $S3_INPUT_1"
echo "  Output: $S3_OUTPUT_1"
echo ""

# --- Job 2: bedrock_output_datasets_prompt_optimized ---
JOB_NAME_2="autoqa-batch-prediction-extractor-optimized"
S3_INPUT_2="s3://your-s3-bucket/autoqa-project/bedrock_output_datasets_prompt_optimized/"
S3_OUTPUT_2="s3://your-s3-bucket/autoqa-project/results/predictions_optimized/"

aws glue create-job \
  --name "$JOB_NAME_2" \
  --role "$ROLE_ARN" \
  --command '{
    "Name": "glueetl",
    "ScriptLocation": "'"$SCRIPT_S3"'",
    "PythonVersion": "3"
  }' \
  --glue-version "4.0" \
  --number-of-workers 10 \
  --worker-type "G.1X" \
  --default-arguments '{
    "--S3_INPUT_PREFIX": "'"$S3_INPUT_2"'",
    "--S3_OUTPUT_PATH": "'"$S3_OUTPUT_2"'",
    "--job-language": "python",
    "--enable-metrics": "true",
    "--enable-continuous-cloudwatch-log": "true"
  }' 2>/dev/null || echo "Job $JOB_NAME_2 already exists, skipping creation."

aws glue start-job-run \
  --job-name "$JOB_NAME_2" \
  --arguments '{
    "--S3_INPUT_PREFIX": "'"$S3_INPUT_2"'",
    "--S3_OUTPUT_PATH": "'"$S3_OUTPUT_2"'"
  }'

echo "Job 2 ($JOB_NAME_2) started."
echo "  Input:  $S3_INPUT_2"
echo "  Output: $S3_OUTPUT_2"
echo ""

echo "Monitor jobs at:"
echo "  https://console.aws.amazon.com/glue/home?region=us-east-1#/v2/etl-configuration/jobs/runs/$JOB_NAME_1"
echo "  https://console.aws.amazon.com/glue/home?region=us-east-1#/v2/etl-configuration/jobs/runs/$JOB_NAME_2"
