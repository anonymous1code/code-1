"""
AWS Glue ETL Job: Extract Predictions from Bedrock/SageMaker Batch Inference Output

Reads .jsonl.out files from S3 (produced by run_bedrock_batch_inference.py),
extracts Yes/No predictions, and writes parts-json output that can be consumed
by evaluate_batch_outputs.py.

Output format (one JSON object per line, written as Spark partitioned JSON):
    {"sample": <int>, "prediction": "yes"/"no", "batch_path": "...", "source_file": "..."}

Glue job arguments:
    --S3_INPUT_PREFIX   S3 path containing .jsonl.out files
                        e.g. s3://your-s3-bucket/autoqa-project/bedrock_output_datasets/
    --S3_OUTPUT_PATH    S3 path for output parts-json files
                        e.g. s3://your-s3-bucket/autoqa-project/results/predictions/

Then evaluate locally:
    python bedrock/evaluate_batch_outputs.py \\
        --output-dir /local/path/to/downloaded/predictions \\
        --dataset /path/to/vcape-s-20k
"""

import re
import sys
import json

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
)

# ---------------------------------------------------------------------------
# Glue / Spark setup
# ---------------------------------------------------------------------------
args = getResolvedOptions(sys.argv, ["JOB_NAME", "S3_INPUT_PREFIX", "S3_OUTPUT_PATH"])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

S3_INPUT_PREFIX = args["S3_INPUT_PREFIX"].rstrip("/")
S3_OUTPUT_PATH = args["S3_OUTPUT_PATH"].rstrip("/")

print(f"S3 Input Prefix: {S3_INPUT_PREFIX}")
print(f"S3 Output Path:  {S3_OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# UDFs for prediction extraction
# ---------------------------------------------------------------------------

def extract_prediction(content):
    """
    Extract the final Yes/No prediction from model output.
    The model outputs <reasoning>...</reasoning> followed by 'Yes' or 'No'.
    """
    if not content:
        return None

    result = content
    match = "a"  # dummy to enter loop
    while match:
        match = re.search(r"</reasoning>\s*(.*)", result, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).strip().lower()
        else:
            return result.lower()

    return "no"


def extract_record_id(record_id_str):
    """Extract the integer sample index from a record_id like 'sample_123'."""
    if not record_id_str:
        return None
    m = re.search(r"sample_(\d+)", str(record_id_str))
    return int(m.group(1)) if m else None


def extract_content_from_json(json_str):
    """
    Parse a single .jsonl.out line and extract (record_id, content_text).
    Supports both SageMaker and Bedrock output formats.
    Returns (sample_index, content_text) or (None, None) on failure.
    """
    try:
        record = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None, None

    # -- Extract record_id --
    record_id = ""
    # SageMaker: payload.record_id
    try:
        record_id = record.get("payload", {}).get("record_id", "")
    except (AttributeError, TypeError):
        pass
    # Bedrock: recordId
    if not record_id:
        record_id = record.get("recordId", "")
    if not record_id:
        record_id = record.get("record_id", "")

    sample_idx = extract_record_id(record_id)
    if sample_idx is None:
        return None, None

    # -- Extract content text --
    content = ""

    # SageMaker format: SageMakerOutput.choices[0].message.content
    try:
        choices = record.get("SageMakerOutput", {}).get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
    except (KeyError, IndexError, TypeError, AttributeError):
        pass

    # Bedrock format: modelOutput.content[].text
    if not content:
        try:
            model_output = record.get("modelOutput", {})
            if isinstance(model_output, str):
                model_output = json.loads(model_output)
            output_content = model_output.get("content", [])
            if isinstance(output_content, list):
                for block in output_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content = block.get("text", "")
                        break
                    elif isinstance(block, dict) and "text" in block:
                        content = block.get("text", "")
                        break
        except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError):
            pass

    # Skip error records
    if not content:
        error = record.get("error")
        if error:
            return None, None

    return sample_idx, content


def process_line(json_str, source_file, s3_input_prefix):
    """
    Process a single JSONL line and return a dict for the output row,
    or None if the line cannot be parsed.
    """
    sample_idx, content = extract_content_from_json(json_str)
    if sample_idx is None:
        return None

    prediction = extract_prediction(content)
    if prediction is None:
        return None

    # Derive batch_path from source_file for dataset name extraction.
    #
    # evaluate_batch_outputs._extract_dataset_name() does:
    #     parts = batch_path.split("/")
    #     return parts[1]  (the dataset name)
    #
    # So batch_path must look like: "bedrock_output_datasets/vcape-r-20k/..."
    # where parts[1] = dataset name.
    #
    # source_file looks like:
    #   s3://bucket/autoqa-project/bedrock_output_datasets/vcape-r-20k/model/batch_0/.../file.jsonl.out
    #   s3://bucket/autoqa-project/bedrock_output_datasets_prompt_optimized/vcape-r-20k/model/batch_0/...
    # S3_INPUT_PREFIX looks like:
    #   s3://bucket/autoqa-project/bedrock_output_datasets/
    #   s3://bucket/autoqa-project/bedrock_output_datasets_prompt_optimized/
    #
    # We strip S3_INPUT_PREFIX from source_file to get the relative path
    # (e.g. "vcape-r-20k/model/batch_0/...") then prepend the folder name.
    prefix = s3_input_prefix.rstrip("/") + "/"
    # Extract folder name from the prefix (e.g. "bedrock_output_datasets" or
    # "bedrock_output_datasets_prompt_optimized")
    folder = s3_input_prefix.rstrip("/").split("/")[-1]

    if source_file.startswith(prefix):
        relative_path = source_file[len(prefix):]
    else:
        # Fallback: find the folder in the source_file path and take everything after it
        m = re.search(re.escape(folder) + r"/(.*)", source_file)
        if m:
            relative_path = m.group(1)
        else:
            # Last resort: use everything after s3://bucket/project/
            m2 = re.search(r"s3://[^/]+/[^/]+/(.*)", source_file)
            relative_path = m2.group(1) if m2 else source_file

    batch_path = folder + "/" + relative_path

    return {
        "sample": sample_idx,
        "prediction": prediction,
        "batch_path": batch_path,
        "source_file": source_file,
    }


# ---------------------------------------------------------------------------
# Register UDF
# ---------------------------------------------------------------------------

output_schema = StructType([
    StructField("sample", IntegerType(), True),
    StructField("prediction", StringType(), True),
    StructField("batch_path", StringType(), True),
    StructField("source_file", StringType(), True),
])

@F.udf(output_schema)
def process_line_udf(value, source_file):
    result = process_line(value, source_file, S3_INPUT_PREFIX)
    if result is None:
        return None
    return (result["sample"], result["prediction"], result["batch_path"], result["source_file"])


# ---------------------------------------------------------------------------
# Main ETL logic
# ---------------------------------------------------------------------------

# 1. Read .jsonl.out files recursively from the S3 prefix.
#    Use pathGlobFilter to only read .jsonl.out files (avoids loading huge
#    input JSONL files into memory), combined with recursiveFileLookup.
print(f"Reading .jsonl.out files recursively from: {S3_INPUT_PREFIX}")

raw_df = (
    spark.read
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.jsonl.out")
    .text(S3_INPUT_PREFIX)
    .withColumn("source_file", F.input_file_name())
)

# 2. Process each line to extract sample, prediction, batch_path, source_file
processed_df = (
    raw_df
    .withColumn("parsed", process_line_udf(F.col("value"), F.col("source_file")))
    .filter(F.col("parsed").isNotNull())
    .select(
        F.col("parsed.sample").alias("sample"),
        F.col("parsed.prediction").alias("prediction"),
        F.col("parsed.batch_path").alias("batch_path"),
        F.col("parsed.source_file").alias("source_file"),
    )
)

# Cache to avoid recomputing the UDF pipeline multiple times
processed_df.cache()

# 3. Write output as JSON (parts-json format)
#    Don't coalesce(1) — it forces all data to a single executor and can OOM.
#    evaluate_batch_outputs.py reads all *.json files in a directory, so
#    multiple part files are fine.
print(f"Writing predictions to: {S3_OUTPUT_PATH}")
(
    processed_df
    .write
    .mode("overwrite")
    .json(S3_OUTPUT_PATH)
)

# 4. Print stats (after write, using cached DF)
valid_count = processed_df.count()
print(f"Valid predictions written: {valid_count}")

print("Prediction distribution:")
processed_df.groupBy("prediction").count().show()

print("Dataset distribution (from batch_path):")
processed_df.withColumn(
    "dataset_name",
    F.split(F.col("batch_path"), "/").getItem(1)
).groupBy("dataset_name").count().show(truncate=False)

processed_df.unpersist()
print(f"Done! Predictions written to {S3_OUTPUT_PATH}")

job.commit()
