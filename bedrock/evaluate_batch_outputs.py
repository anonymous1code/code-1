#!/usr/bin/env python3
"""
Evaluate batch inference outputs from .jsonl.out files.

Reads all .jsonl.out files in a given folder, extracts model predictions (Yes/No),
loads ground truth from a HuggingFace dataset, and computes:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

Usage:
    python evaluate_batch_outputs.py \
        --output-dir bedrock/outputs/QWEN3-8B/vcape-s \
        --dataset /path/to/vcape-s-20k

    # If dataset is on S3, download first or point to local cache:
    python evaluate_batch_outputs.py \
        --output-dir bedrock/outputs/QWEN3-8B/vcape-s \
        --dataset /tmp/vcape-dataset-cache/vcape-s-20k
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
except ImportError:
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
    sys.exit(1)


def extract_prediction(content: str) -> Optional[str]:
    """
    Extract the final Yes/No prediction from the model output.
    The model outputs <reasoning>...</reasoning> followed by 'Yes' or 'No'.
    """
    if not content:
        return None
    # Get the last meaningful word after </reasoning> or at end of content
    # Strip whitespace and look for Yes/No at the end
    text = content.strip()
    # Try to find text after </reasoning>
    match = "a"
    result = content
    while match:
        match = re.search(r"</reasoning>\s*(.*)", result, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).strip().lower().replace("'", "")
        else:
            return "yes" if "yes" in result else "no"
        
    return "no"


def _is_parts_json_file(fpath: Path) -> bool:
    """
    Detect if a file is in the "parts-json" format produced by Spark / Glue
    extraction jobs.  Each line looks like:
        {"sample": 0, "prediction": "yes", "batch_path": "...", "source_file": "..."}
    """
    try:
        with open(fpath, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                return False
            record = json.loads(first_line)
            return "sample" in record and "prediction" in record
    except (json.JSONDecodeError, OSError):
        return False


def _extract_dataset_name(batch_path: str) -> str:
    """
    Extract dataset name from batch_path.
    e.g. 'bedrock_output_datasets/vcape-r-20k/us.anthropic...' -> 'vcape-r-20k'
    """
    parts = batch_path.split("/")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def _extract_model_name(batch_path: str) -> str:
    """
    Extract model name from batch_path.
    e.g. 'bedrock_output_datasets/vcape-r-20k/us.anthropic.claude-3-7-sonnet-20250219-/...'
         -> 'us.anthropic.claude-3-7-sonnet-20250219-'
    """
    parts = batch_path.split("/")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


def _extract_batch_number(source_file: str) -> int:
    """Extract batch number from source_file path. e.g. '.../batch_3/...' or 'batch_3.jsonl.out' -> 3"""
    m = re.search(r'batch_(\d+)', source_file)
    return int(m.group(1)) if m else 0


def parse_parts_json_files(files: List[Path]) -> Dict[str, Dict[int, str]]:
    """
    Parse parts-json files where each line is:
        {"sample": <int>, "prediction": "yes"/"no", "batch_path": "...", "source_file": "..."}

    The 'sample' field may be a Spark monotonically_increasing_id (not the
    real dataset index). If indices look like real dataset indices (the 'sample'
    field matches a sample_N pattern or fits within expected range), they are
    used directly.  Otherwise, we reconstruct real indices by sorting records
    by (batch_number, spark_id) within each dataset.

    Returns dict mapping dataset_name -> {sample_index -> prediction}.
    """
    # First pass: collect all records grouped by dataset
    records_by_dataset: Dict[str, List[dict]] = {}
    parse_errors = 0

    for fpath in files:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                sample_idx = record.get("sample")
                prediction = record.get("prediction", "").lower().strip()

                if sample_idx is None or prediction not in ("yes", "no"):
                    parse_errors += 1
                    continue

                batch_path = record.get("batch_path", "")
                dataset_name = _extract_dataset_name(batch_path) if batch_path else "unknown"

                if dataset_name not in records_by_dataset:
                    records_by_dataset[dataset_name] = []
                records_by_dataset[dataset_name].append({
                    "spark_id": int(sample_idx),
                    "prediction": prediction,
                    "source_file": record.get("source_file", ""),
                })

    if parse_errors > 0:
        print(f"Warning: {parse_errors} records could not be parsed in parts-json files")

    # Second pass: for each dataset, decide if indices need reconstruction
    predictions_by_dataset: Dict[str, Dict[int, str]] = {}

    for ds_name, records in records_by_dataset.items():
        spark_ids = [r["spark_id"] for r in records]
        max_id = max(spark_ids)
        n_records = len(records)

        # Heuristic: if max spark_id is much larger than record count,
        # the IDs are Spark partition offsets and need reconstruction.
        # Real dataset indices would be in range [0, N).
        needs_reindex = max_id > n_records * 2

        if needs_reindex:
            # Sort by (batch_number, spark_id) to recover original order
            for r in records:
                r["batch_num"] = _extract_batch_number(r["source_file"])
            records.sort(key=lambda r: (r["batch_num"], r["spark_id"]))

            preds = {}
            for seq_idx, r in enumerate(records):
                preds[seq_idx] = r["prediction"]

            print(f"  [{ds_name}] Re-indexed {n_records} records "
                  f"(Spark IDs {min(spark_ids)}-{max_id} → 0-{n_records-1})")
        else:
            # IDs look like real dataset indices, use as-is
            preds = {r["spark_id"]: r["prediction"] for r in records}

        predictions_by_dataset[ds_name] = preds

    return predictions_by_dataset


def parse_jsonl_out_files(files: List[Path]) -> Dict[int, str]:
    """
    Parse .jsonl.out files (SageMaker / Bedrock raw batch output).
    Returns dict mapping sample_index -> prediction ('yes'/'no').
    """
    predictions = {}
    parse_errors = 0

    for fpath in files:
        with open(fpath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                # Extract record_id -> sample index
                # Support multiple formats:
                #   SageMaker: {"payload": {"record_id": "sample_123"}, ...}
                #   Bedrock:   {"recordId": "sample_123", ...}
                record_id = record.get("payload", {}).get("record_id", "")
                if not record_id:
                    record_id = record.get("recordId", "")
                if not record_id:
                    record_id = record.get("record_id", "")

                match = re.search(r'sample_(\d+)', record_id)
                if not match:
                    parse_errors += 1
                    continue
                sample_idx = int(match.group(1))

                # Extract model prediction from response content
                content = ""
                try:
                    choices = record.get("SageMakerOutput", {}).get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                except (KeyError, IndexError, TypeError):
                    pass

                # Try Bedrock batch output format:
                # {"modelOutput": {"content": [{"text": "..."}], ...}, ...}
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
                    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                        pass

                # Skip records with errors (Bedrock format)
                if not content and "error" in record and record["error"]:
                    parse_errors += 1
                    continue

                prediction = extract_prediction(content)
                if prediction is not None:
                    predictions[sample_idx] = prediction
                else:
                    parse_errors += 1

    if parse_errors > 0:
        print(f"Warning: {parse_errors} records could not be parsed in jsonl.out files")

    return predictions


def _discover_output_files(output_dir: str) -> Tuple[List[Path], List[Path]]:
    """Discover parts-json and jsonl.out files in output_dir."""
    output_path = Path(output_dir)

    # Parts-json (*.json files, including part-* files)
    json_files = sorted(output_path.glob("*.json"))
    if not json_files:
        json_files = sorted(output_path.rglob("part-*.json"))

    parts_json_files = []
    for f in json_files:
        if f.name == "metrics.json":
            continue
        if _is_parts_json_file(f):
            parts_json_files.append(f)

    # .jsonl.out files
    jsonl_files = sorted(output_path.glob("*.jsonl.out"))
    if not jsonl_files:
        jsonl_files = sorted(output_path.glob("*.jsonl.out.*"))
    if not jsonl_files:
        jsonl_files = sorted(output_path.glob("*.jsonl"))
    if not jsonl_files:
        jsonl_files = sorted(output_path.rglob("*.jsonl.out"))
    if not jsonl_files:
        jsonl_files = sorted(output_path.rglob("*.jsonl.out.*"))
    if not jsonl_files:
        jsonl_files = sorted(output_path.rglob("*.jsonl"))

    return parts_json_files, jsonl_files


def parse_output_files(output_dir: str) -> Dict[int, str]:
    """
    Parse batch output files in the given directory (flat — no dataset split).
    Returns a dict mapping sample_index -> prediction ('yes'/'no').
    """
    parts_json_files, jsonl_files = _discover_output_files(output_dir)

    if not parts_json_files and not jsonl_files:
        print(f"ERROR: No output files found in {output_dir}")
        sys.exit(1)

    predictions = {}
    if parts_json_files:
        print(f"Found {len(parts_json_files)} parts-json file(s) in {output_dir}")
        by_ds = parse_parts_json_files(parts_json_files)
        for ds_preds in by_ds.values():
            predictions.update(ds_preds)
        print(f"  Parsed {len(predictions)} predictions from parts-json")

    if jsonl_files:
        print(f"Found {len(jsonl_files)} .jsonl.out file(s) in {output_dir}")
        jsonl_preds = parse_jsonl_out_files(jsonl_files)
        before = len(predictions)
        for k, v in jsonl_preds.items():
            if k not in predictions:
                predictions[k] = v
        print(f"  Parsed {len(jsonl_preds)} predictions from jsonl.out ({len(predictions) - before} new)")

    print(f"Total predictions parsed: {len(predictions)}")
    return predictions


def parse_output_files_by_dataset(output_dir: str) -> Dict[str, Dict[int, str]]:
    """
    Parse batch output files and split predictions by dataset name.
    Dataset name is extracted from the batch_path field in parts-json records
    (e.g. 'vcape-s-20k', 'vcape-r-20k').

    For .jsonl.out files without batch_path, they are grouped under the parent
    directory name (e.g. 'vcape-s' from outputs/QWEN3-8B/vcape-s/).

    Returns: { dataset_name: {sample_index: prediction} }
    """
    parts_json_files, jsonl_files = _discover_output_files(output_dir)

    if not parts_json_files and not jsonl_files:
        print(f"ERROR: No output files found in {output_dir}")
        sys.exit(1)

    by_dataset: Dict[str, Dict[int, str]] = {}

    if parts_json_files:
        print(f"Found {len(parts_json_files)} parts-json file(s) in {output_dir}")
        pj = parse_parts_json_files(parts_json_files)
        for ds_name, preds in pj.items():
            by_dataset[ds_name] = preds
            print(f"  [{ds_name}] {len(preds)} predictions from parts-json")

    if jsonl_files:
        print(f"Found {len(jsonl_files)} .jsonl.out file(s) in {output_dir}")
        jsonl_preds = parse_jsonl_out_files(jsonl_files)
        # Infer dataset name from directory structure
        ds_name = Path(output_dir).name  # e.g. 'vcape-s'
        if ds_name not in by_dataset:
            by_dataset[ds_name] = {}
        before = len(by_dataset[ds_name])
        for k, v in jsonl_preds.items():
            if k not in by_dataset[ds_name]:
                by_dataset[ds_name][k] = v
        added = len(by_dataset[ds_name]) - before
        print(f"  [{ds_name}] {len(jsonl_preds)} predictions from jsonl.out ({added} new)")

    for ds_name, preds in sorted(by_dataset.items()):
        print(f"Dataset '{ds_name}': {len(preds)} total predictions")
    return by_dataset


def _parse_parts_json_by_dataset_and_model(
    files: List[Path],
) -> Dict[str, Dict[str, Dict[int, str]]]:
    """
    Parse parts-json files and split predictions by (dataset, model).

    Returns: { dataset_name: { model_name: {sample_index: prediction} } }
    """
    # Collect records grouped by (dataset, model)
    records_by_key: Dict[str, Dict[str, List[dict]]] = {}
    parse_errors = 0

    for fpath in files:
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue

                sample_idx = record.get("sample")
                prediction = record.get("prediction", "").lower().strip()

                if sample_idx is None or prediction not in ("yes", "no"):
                    parse_errors += 1
                    continue

                batch_path = record.get("batch_path", "")
                dataset_name = _extract_dataset_name(batch_path) if batch_path else "unknown"
                model_name = _extract_model_name(batch_path) if batch_path else "unknown"

                if dataset_name not in records_by_key:
                    records_by_key[dataset_name] = {}
                if model_name not in records_by_key[dataset_name]:
                    records_by_key[dataset_name][model_name] = []

                records_by_key[dataset_name][model_name].append({
                    "spark_id": int(sample_idx),
                    "prediction": prediction,
                    "source_file": record.get("source_file", ""),
                })

    if parse_errors > 0:
        print(f"Warning: {parse_errors} records could not be parsed in parts-json files")

    # Resolve indices per (dataset, model)
    result: Dict[str, Dict[str, Dict[int, str]]] = {}
    for ds_name, models in records_by_key.items():
        result[ds_name] = {}
        for model_name, records in models.items():
            spark_ids = [r["spark_id"] for r in records]
            max_id = max(spark_ids)
            n_records = len(records)
            needs_reindex = max_id > n_records * 2

            if needs_reindex:
                for r in records:
                    r["batch_num"] = _extract_batch_number(r["source_file"])
                records.sort(key=lambda r: (r["batch_num"], r["spark_id"]))
                preds = {seq_idx: r["prediction"] for seq_idx, r in enumerate(records)}
                print(f"  [{ds_name}/{model_name}] Re-indexed {n_records} records")
            else:
                preds = {r["spark_id"]: r["prediction"] for r in records}

            result[ds_name][model_name] = preds

    return result


def parse_output_files_by_dataset_and_model(
    output_dir: str,
) -> Dict[str, Dict[str, Dict[int, str]]]:
    """
    Parse batch output files and split predictions by (dataset, model).

    Returns: { dataset_name: { model_name: {sample_index: prediction} } }
    """
    parts_json_files, jsonl_files = _discover_output_files(output_dir)

    if not parts_json_files and not jsonl_files:
        print(f"ERROR: No output files found in {output_dir}")
        sys.exit(1)

    by_ds_model: Dict[str, Dict[str, Dict[int, str]]] = {}

    if parts_json_files:
        print(f"Found {len(parts_json_files)} parts-json file(s) in {output_dir}")
        by_ds_model = _parse_parts_json_by_dataset_and_model(parts_json_files)
        for ds_name, models in sorted(by_ds_model.items()):
            for model_name, preds in sorted(models.items()):
                print(f"  [{ds_name} / {model_name}] {len(preds)} predictions")

    if jsonl_files:
        print(f"Found {len(jsonl_files)} .jsonl.out file(s) in {output_dir}")
        jsonl_preds = parse_jsonl_out_files(jsonl_files)
        ds_name = Path(output_dir).name
        model_name = "unknown"
        if ds_name not in by_ds_model:
            by_ds_model[ds_name] = {}
        if model_name not in by_ds_model[ds_name]:
            by_ds_model[ds_name][model_name] = {}
        for k, v in jsonl_preds.items():
            if k not in by_ds_model[ds_name][model_name]:
                by_ds_model[ds_name][model_name][k] = v if v in ["yes", "no"] else "yes" if "yes" in v else "no"
        print(f"  [{ds_name}/{model_name}] {len(jsonl_preds)} predictions from jsonl.out")

    # Summary
    for ds_name, models in sorted(by_ds_model.items()):
        for model_name, preds in sorted(models.items()):
            print(f"  {ds_name} / {model_name}: {len(preds)} predictions")

    return by_ds_model


def load_ground_truth(dataset_path: str) -> Tuple[Dict[int, bool], Dict[int, str]]:
    """
    Load ground truth labels from a HuggingFace dataset.
    The dataset has a 'label' column with 'accept'/'reject' values and optionally
    a 'rejection_reason' column.
    Returns:
        - dict mapping sample_index -> bool (True = accept/Yes, False = reject/No)
        - dict mapping sample_index -> rejection_reason string (empty string if accepted)
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("ERROR: 'datasets' library required for loading ground truth.")
        print("Install with: pip install datasets")
        sys.exit(1)

    print(f"Loading dataset from: {dataset_path}")
    ds = Dataset.load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(ds)} samples, columns: {ds.column_names}")

    # Determine which column holds the ground truth label
    if "label" in ds.column_names:
        label_col = "label"
    elif "is_generation_successful" in ds.column_names:
        label_col = "is_generation_successful"
    else:
        print(f"ERROR: Cannot find ground truth column. Available: {ds.column_names}")
        sys.exit(1)

    # Check for rejection_reason column
    has_rejection_reason = "rejection_reason" in ds.column_names

    gt = {}
    rejection_reasons = {}
    for idx in range(len(ds)):
        val = ds[idx][label_col]
        if isinstance(val, bool):
            gt[idx] = val
        elif isinstance(val, str):
            v = val.lower().strip()
            # 'accept'/'accepted' -> True, 'reject'/'rejected' -> False  (label column)
            # 'true' -> True, 'false' -> False  (is_generation_successful column)
            gt[idx] = v in ("accept", "accepted", "true", "yes", "1")
        else:
            gt[idx] = bool(val)

        if has_rejection_reason:
            reason = ds[idx]["rejection_reason"]
            rejection_reasons[idx] = str(reason).strip() if reason else ""
        else:
            rejection_reasons[idx] = ""

    pos = sum(gt.values())
    print(f"Ground truth: {pos} positive (accept), {len(gt) - pos} negative (reject)")
    if has_rejection_reason:
        reasons_counter = Counter(v for v in rejection_reasons.values() if v)
        print(f"Rejection reasons found: {len(reasons_counter)} unique values")
        for reason, count in reasons_counter.most_common():
            print(f"    {reason}: {count}")
    return gt, rejection_reasons


def compute_metrics(
    predictions: Dict[int, str],
    ground_truth: Dict[int, bool],
) -> Dict:
    """
    Compute evaluation metrics.
    Positive class: Yes (consistent / generation successful)
    Negative class: No (inconsistent / generation not successful)
    """
    # Align predictions with ground truth
    common_indices = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    if not common_indices:
        print("ERROR: No overlapping indices between predictions and ground truth!")
        sys.exit(1)

    missing_pred = set(ground_truth.keys()) - set(predictions.keys())
    missing_gt = set(predictions.keys()) - set(ground_truth.keys())

    if missing_pred:
        print(f"Warning: {len(missing_pred)} samples have ground truth but no prediction")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} samples have predictions but no ground truth")

    y_true = []  # bool: True = successful/consistent (Yes)
    y_pred = []  # bool: True = predicted Yes

    for idx in common_indices:
        y_true.append(ground_truth[idx])
        y_pred.append(predictions[idx] == 'yes')

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    # Also compute metrics for the negative class (No)
    precision_neg, recall_neg, f1_neg, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=False, zero_division=0
    )

    # Prediction distribution
    pred_counter = Counter(predictions.values())

    return {
        "total_evaluated": len(common_indices),
        "total_predictions": len(predictions),
        "total_ground_truth": len(ground_truth),
        "accuracy": accuracy,
        "precision_positive (Yes)": precision,
        "recall_positive (Yes)": recall,
        "f1_positive (Yes)": f1,
        "precision_negative (No)": precision_neg,
        "recall_negative (No)": recall_neg,
        "f1_negative (No)": f1_neg,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "gt_positive_count": int(sum(y_true)),
        "gt_negative_count": int(len(y_true) - sum(y_true)),
        "pred_yes_count": int(pred_counter.get('yes', 0)),
        "pred_no_count": int(pred_counter.get('no', 0)),
    }


def compute_metrics_per_rejection_reason(
    predictions: Dict[int, str],
    ground_truth: Dict[int, bool],
    rejection_reasons: Dict[int, str],
) -> Dict[str, Dict]:
    """
    Compute evaluation metrics broken down by rejection reason.
    
    Groups samples by their rejection reason in the ground truth dataset and
    computes accuracy, precision, recall, and F1 for each group.
    
    Samples with label='accept' (gt=True) are grouped under 'accepted'.
    Samples with label='reject' are grouped by their rejection_reason value.
    
    Returns dict mapping rejection_reason -> metrics dict.
    """
    # Align predictions with ground truth
    common_indices = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    if not common_indices:
        return {}

    # Group indices by rejection reason
    groups: Dict[str, List[int]] = {}
    for idx in common_indices:
        if ground_truth[idx]:  # accepted sample
            reason = "accepted"
        else:
            reason = rejection_reasons.get(idx, "").strip()
            if not reason:
                reason = "rejected (no reason)"
        if reason not in groups:
            groups[reason] = []
        groups[reason].append(idx)

    results = {}
    for reason, indices in sorted(groups.items(), key=lambda x: -len(x[1])):
        y_true = [ground_truth[idx] for idx in indices]
        y_pred = [predictions[idx] == 'yes' for idx in indices]

        n_samples = len(indices)
        n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = n_correct / n_samples if n_samples > 0 else 0.0

        # For accepted samples: correct prediction is 'yes'
        # For rejected samples: correct prediction is 'no'
        if reason == "accepted":
            # All gt are True (accept), so we measure how many we predicted 'yes'
            n_predicted_correctly = sum(1 for p in y_pred if p)  # predicted yes = correct
            recall_for_group = n_predicted_correctly / n_samples if n_samples > 0 else 0.0
            results[reason] = {
                "count": n_samples,
                "accuracy": accuracy,
                "correct_predictions": n_correct,
                "predicted_yes": sum(1 for p in y_pred if p),
                "predicted_no": sum(1 for p in y_pred if not p),
                "recall": recall_for_group,  # How many accepted were correctly predicted as yes
            }
        else:
            # All gt are False (reject), so we measure how many we predicted 'no'
            n_predicted_correctly = sum(1 for p in y_pred if not p)  # predicted no = correct
            recall_for_group = n_predicted_correctly / n_samples if n_samples > 0 else 0.0
            results[reason] = {
                "count": n_samples,
                "accuracy": accuracy,
                "correct_predictions": n_correct,
                "predicted_yes": sum(1 for p in y_pred if p),
                "predicted_no": sum(1 for p in y_pred if not p),
                "recall": recall_for_group,  # How many rejected were correctly predicted as no
            }

    return results


def print_metrics_per_rejection_reason(metrics_by_reason: Dict[str, Dict]):
    """Pretty print per-rejection-reason metrics."""
    if not metrics_by_reason:
        return

    print("\n" + "=" * 80)
    print("          PERFORMANCE PER REJECTION REASON")
    print("=" * 80)

    # Print header
    print(f"\n  {'Rejection Reason':<40} {'Count':>6} {'Accuracy':>9} {'Recall':>8} {'Pred Yes':>9} {'Pred No':>8}")
    print(f"  {'─' * 40} {'─' * 6} {'─' * 9} {'─' * 8} {'─' * 9} {'─' * 8}")

    # Print accepted first, then rejected reasons sorted by count
    if "accepted" in metrics_by_reason:
        m = metrics_by_reason["accepted"]
        print(f"  {'✓ accepted':<40} {m['count']:>6} {m['accuracy']:>8.4f} {m['recall']:>8.4f} {m['predicted_yes']:>9} {m['predicted_no']:>8}")

    print(f"  {'─' * 40} {'─' * 6} {'─' * 9} {'─' * 8} {'─' * 9} {'─' * 8}")

    for reason, m in sorted(
        ((r, m) for r, m in metrics_by_reason.items() if r != "accepted"),
        key=lambda x: -x[1]["count"],
    ):
        # Truncate long reason names
        display_reason = reason[:38] + ".." if len(reason) > 40 else reason
        print(f"  ✗ {display_reason:<38} {m['count']:>6} {m['accuracy']:>8.4f} {m['recall']:>8.4f} {m['predicted_yes']:>9} {m['predicted_no']:>8}")

    # Summary
    total_rejected = sum(m["count"] for r, m in metrics_by_reason.items() if r != "accepted")
    total_rejected_correct = sum(m["correct_predictions"] for r, m in metrics_by_reason.items() if r != "accepted")
    total_accepted = metrics_by_reason.get("accepted", {}).get("count", 0)
    total_accepted_correct = metrics_by_reason.get("accepted", {}).get("correct_predictions", 0)

    print(f"\n  {'─' * 76}")
    print(f"  Summary:")
    print(f"    Accepted samples: {total_accepted_correct}/{total_accepted} correctly predicted "
          f"({total_accepted_correct/total_accepted*100:.1f}%)" if total_accepted > 0 else "")
    print(f"    Rejected samples: {total_rejected_correct}/{total_rejected} correctly predicted "
          f"({total_rejected_correct/total_rejected*100:.1f}%)" if total_rejected > 0 else "")
    print("=" * 80)


def print_metrics(metrics: Dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("          EVALUATION METRICS")
    print("=" * 60)

    print(f"\n  Samples evaluated:     {metrics['total_evaluated']}")
    print(f"  Total predictions:     {metrics['total_predictions']}")
    print(f"  Total ground truth:    {metrics['total_ground_truth']}")

    print(f"\n  Ground Truth Distribution:")
    print(f"    Positive (Yes):      {metrics['gt_positive_count']}")
    print(f"    Negative (No):       {metrics['gt_negative_count']}")

    print(f"\n  Prediction Distribution:")
    print(f"    Predicted Yes:       {metrics['pred_yes_count']}")
    print(f"    Predicted No:        {metrics['pred_no_count']}")

    print(f"\n  {'─' * 40}")
    print(f"  Accuracy:              {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  {'─' * 40}")
    print(f"  Positive class (Yes = consistent):")
    print(f"    Precision:           {metrics['precision_positive (Yes)']:.4f}")
    print(f"    Recall:              {metrics['recall_positive (Yes)']:.4f}")
    print(f"    F1 Score:            {metrics['f1_positive (Yes)']:.4f}")
    print(f"  {'─' * 40}")
    print(f"  Negative class (No = inconsistent):")
    print(f"    Precision:           {metrics['precision_negative (No)']:.4f}")
    print(f"    Recall:              {metrics['recall_negative (No)']:.4f}")
    print(f"    F1 Score:            {metrics['f1_negative (No)']:.4f}")
    print(f"  {'─' * 40}")

    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted No   Predicted Yes")
    print(f"    Actual No       {metrics['true_negatives']:>10}     {metrics['false_positives']:>10}")
    print(f"    Actual Yes      {metrics['false_negatives']:>10}     {metrics['true_positives']:>10}")

    print("\n" + "=" * 60)


def print_metrics_header(dataset_name: str, model_name: str = None):
    """Print a header for a dataset (and optionally model) evaluation section."""
    print("\n" + "#" * 60)
    if model_name:
        print(f"  DATASET: {dataset_name}")
        print(f"  MODEL:   {model_name}")
    else:
        print(f"  DATASET: {dataset_name}")
    print("#" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate batch inference outputs (Precision, Recall, Accuracy, F1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset (legacy):
  python evaluate_batch_outputs.py \\
      --output-dir bedrock/outputs/QWEN3-8B/vcape-s \\
      --dataset /path/to/vcape-s-20k

  # Multi-dataset from parts-json (evaluates both s and r):
  python evaluate_batch_outputs.py \\
      --output-dir bedrock/outputs \\
      --datasets vcape-s-20k=/path/to/vcape-s-20k vcape-r-20k=/path/to/vcape-r-20k

  # Save metrics per dataset:
  python evaluate_batch_outputs.py \\
      --output-dir bedrock/outputs \\
      --datasets vcape-s-20k=/path/to/vcape-s-20k vcape-r-20k=/path/to/vcape-r-20k \\
      --save-json bedrock/outputs/metrics.json
""",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory containing batch output files (.jsonl.out or parts-json).",
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=None,
        help="Dataset path(s). Accepts either:\n"
             "  - A single path: /path/to/vcape-s-20k\n"
             "  - Multiple NAME=PATH mappings: vcape-s-20k=/path/to/s vcape-r-20k=/path/to/r",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Alias for --dataset with multiple NAME=PATH mappings.",
    )
    parser.add_argument(
        "--save-json", type=str, default=None,
        help="Optional path to save metrics as JSON file.",
    )
    args = parser.parse_args()

    if not args.dataset and not args.datasets:
        parser.error("Either --dataset or --datasets is required")

    # --- Multi-dataset mode ---
    if args.datasets:
        # Parse NAME=PATH mappings
        dataset_map = {}
        for mapping in args.datasets:
            if "=" not in mapping:
                print(f"ERROR: Invalid dataset mapping '{mapping}'. Expected NAME=PATH")
                sys.exit(1)
            name, path = mapping.split("=", 1)
            dataset_map[name.strip()] = path.strip()

        print(f"Multi-dataset mode: {len(dataset_map)} datasets configured")
        for name, path in dataset_map.items():
            print(f"  {name} -> {path}")

        # Parse predictions split by (dataset, model)
        by_ds_model = parse_output_files_by_dataset_and_model(args.output_dir)

        # Cache ground truth so we don't reload for each model
        gt_cache: Dict[str, Tuple[Dict[int, bool], Dict[int, str]]] = {}

        all_metrics = {}
        for ds_name in sorted(by_ds_model.keys()):
            models = by_ds_model[ds_name]

            # Find matching ground truth dataset
            gt_path = dataset_map.get(ds_name)
            if not gt_path:
                total_preds = sum(len(p) for p in models.values())
                print(f"\nWARNING: No ground truth dataset provided for '{ds_name}' "
                      f"({total_preds} predictions across {len(models)} model(s)). Skipping.")
                print(f"  Available mappings: {list(dataset_map.keys())}")
                continue

            # Load ground truth (cached)
            if ds_name not in gt_cache:
                gt_cache[ds_name] = load_ground_truth(gt_path)
            gt, rejection_reasons = gt_cache[ds_name]

            all_metrics[ds_name] = {}
            for model_name in sorted(models.keys()):
                preds = models[model_name]
                print_metrics_header(ds_name, model_name)
                metrics = compute_metrics(preds, gt)
                print_metrics(metrics)

                # Per-rejection-reason breakdown
                reason_metrics = compute_metrics_per_rejection_reason(preds, gt, rejection_reasons)
                print_metrics_per_rejection_reason(reason_metrics)
                metrics["per_rejection_reason"] = reason_metrics

                all_metrics[ds_name][model_name] = metrics

        # Save combined metrics
        if args.save_json:
            with open(args.save_json, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\nAll metrics saved to: {args.save_json}")
        return

    # --- Single dataset mode (legacy) ---
    predictions = parse_output_files(args.output_dir)
    # --dataset accepts a single path or a list; use the first element
    dataset_path = args.dataset[0] if isinstance(args.dataset, list) else args.dataset
    ground_truth, rejection_reasons = load_ground_truth(dataset_path)
    metrics = compute_metrics(predictions, ground_truth)
    print_metrics(metrics)

    # Per-rejection-reason breakdown
    reason_metrics = compute_metrics_per_rejection_reason(predictions, ground_truth, rejection_reasons)
    print_metrics_per_rejection_reason(reason_metrics)
    metrics["per_rejection_reason"] = reason_metrics

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.save_json}")


if __name__ == "__main__":
    main()

#python evaluate_batch_outputs.py --output-dir outputs --datasets vcape-s-20k=~/tmp/vcape-dataset-cache/vcape-s-20k vcape-r-20k=~/tmp/vcape-dataset-cache/vcape-r-20k --save-json outputs/metrics.json