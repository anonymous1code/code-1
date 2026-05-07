"""
Fine-tuner for Qwen3 AutoQA.

Orchestrates LoRA fine-tuning of Qwen3-VL-8B-Instruct using
PEFT + TRL SFTTrainer on chat-format training examples produced by
:mod:`scripts.training_formatter`.

Usage (as a library):
    from scripts.fine_tuner import run_finetuning

    run_finetuning(
        train_dataset=formatted_examples,
        output_dir="models/qwen3-autoqa-lora/n5",
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multimodal collate function
# ---------------------------------------------------------------------------

def _extract_images_from_messages(messages: list[dict]) -> list[Image.Image]:
    """Pull PIL images out of the inline message content dicts."""
    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                img = item.get("image")
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
    return images


def make_collate_fn(processor):
    """Return a collate function that tokenises chat messages with images.

    The collate function:
    1. Applies the chat template to get the text representation.
    2. Extracts PIL images from the inline message content.
    3. Runs the processor to produce input_ids, pixel_values, etc.
    4. Builds labels (masking pad tokens with -100).
    """

    def collate_fn(examples: list[dict]) -> dict:
        texts = []
        all_images = []

        for ex in examples:
            msgs = ex["messages"]
            # apply_chat_template produces the full text with special tokens
            text = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            texts.append(text.strip())

            images = _extract_images_from_messages(msgs)
            all_images.extend(images)

        # Processor handles both text tokenisation and image encoding.
        # Images are passed as a flat list — the processor matches them
        # to <|image_pad|> placeholders in the tokenised text.
        batch = processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )

        # Build labels: clone input_ids, mask padding with -100
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    return collate_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_finetuning(
    train_dataset: Dataset | list[dict],
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    output_dir: str = "models/qwen3-autoqa-lora/n1",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_checkpointing: bool = True,
) -> str:
    """Fine-tune the base model with LoRA and return the adapter output path.

    Parameters
    ----------
    train_dataset : Dataset | list[dict]
        Training examples in Qwen3-VL chat format.  Each element must
        contain a ``messages`` key with user and assistant turns, as
        produced by :func:`scripts.training_formatter.format_all_examples`.
    base_model : str
        HuggingFace model identifier for the base VLM.
    output_dir : str
        Directory where the PEFT adapter weights will be saved.
    lora_rank : int
        Rank of the LoRA decomposition matrices.
    lora_alpha : int
        Scaling factor for LoRA updates.
    learning_rate : float
        Peak learning rate for the AdamW optimiser.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Per-device training batch size.
    gradient_checkpointing : bool
        If *True*, enable gradient checkpointing to trade compute for
        memory savings.

    Returns
    -------
    str
        Path to the saved adapter directory (*output_dir*).

    Raises
    ------
    torch.cuda.OutOfMemoryError
        Re-raised after printing a suggestion to reduce batch size or
        enable gradient checkpointing.
    """

    # ------------------------------------------------------------------
    # 1. Wrap list[dict] in a HuggingFace Dataset if needed
    # ------------------------------------------------------------------
    if isinstance(train_dataset, list):
        train_dataset = Dataset.from_list(train_dataset)

    # ------------------------------------------------------------------
    # 1b. Split off an eval set (20% or at least 1 sample) for loss
    #     tracking, unless the dataset is too small (≤2 samples).
    # ------------------------------------------------------------------
    eval_dataset = None
    if len(train_dataset) > 2:
        n_eval = max(1, int(len(train_dataset) * 0.2))
        splits = train_dataset.train_test_split(
            test_size=n_eval, seed=42,
        )
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
        logger.info(
            "Train/eval split: %d train, %d eval",
            len(train_dataset), len(eval_dataset),
        )

    # ------------------------------------------------------------------
    # 2. Load base model + processor
    # ------------------------------------------------------------------
    logger.info("Loading base model: %s", base_model)

    # Load the full model onto a single device to avoid meta-device
    # gradient mismatches during backward.
    device_map = {"": "cuda:0"} if torch.cuda.is_available() else {"": "cpu"}

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    processor = AutoProcessor.from_pretrained(base_model)

    # ------------------------------------------------------------------
    # 3. LoRA adapter config
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        r=lora_rank,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # ------------------------------------------------------------------
    # 4. SFTTrainer training arguments
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=learning_rate,
        logging_steps=1,
        logging_strategy="steps",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        save_strategy="epoch",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        # Skip TRL's built-in dataset tokenisation — our custom collate_fn
        # handles multimodal tokenisation (images + text) instead.
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # 5. Build custom collate function for multimodal data
    # ------------------------------------------------------------------
    collate_fn = make_collate_fn(processor)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor,
    )

    try:
        logger.info(
            "Starting training: %d examples, %d epochs, batch_size=%d, "
            "lr=%g, lora_rank=%d, lora_alpha=%d",
            len(train_dataset),
            num_epochs,
            batch_size,
            learning_rate,
            lora_rank,
            lora_alpha,
        )
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print(
            "\n*** CUDA Out-of-Memory Error ***\n"
            "The training run exceeded available GPU memory.\n"
            "Suggestions:\n"
            "  1. Reduce --batch-size to 1 (current: {bs})\n"
            "  2. Enable gradient checkpointing (already on by default)\n".format(
                bs=batch_size,
            ),
            file=sys.stderr,
        )
        raise

    # ------------------------------------------------------------------
    # 7. Save adapter weights (PEFT-compatible format)
    # ------------------------------------------------------------------
    trainer.save_model(output_dir)
    logger.info("Adapter weights saved to %s", output_dir)

    # ------------------------------------------------------------------
    # 8. Save training log history and metrics summary
    # ------------------------------------------------------------------
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    logger.info("Training log saved to %s", log_path)

    # Build a structured metrics summary for easy plotting
    metrics_summary = {
        "config": {
            "base_model": base_model,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "num_train_samples": len(train_dataset),
            "num_eval_samples": len(eval_dataset) if eval_dataset else 0,
        },
        "train_loss": [],
        "eval_loss": [],
        "learning_rate": [],
        "grad_norm": [],
        "mean_token_accuracy": [],
    }

    for entry in trainer.state.log_history:
        step = entry.get("step")
        epoch = entry.get("epoch")
        if "loss" in entry:
            metrics_summary["train_loss"].append({
                "step": step, "epoch": epoch, "value": entry["loss"],
            })
        if "eval_loss" in entry:
            metrics_summary["eval_loss"].append({
                "step": step, "epoch": epoch, "value": entry["eval_loss"],
            })
        if "learning_rate" in entry:
            metrics_summary["learning_rate"].append({
                "step": step, "epoch": epoch, "value": entry["learning_rate"],
            })
        if "grad_norm" in entry:
            metrics_summary["grad_norm"].append({
                "step": step, "epoch": epoch, "value": entry["grad_norm"],
            })
        if "mean_token_accuracy" in entry:
            metrics_summary["mean_token_accuracy"].append({
                "step": step, "epoch": epoch,
                "value": entry["mean_token_accuracy"],
            })

    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Metrics summary saved to %s", metrics_path)

    return output_dir
