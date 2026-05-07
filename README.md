[Data](https://huggingface.co/datasets/anonymous1code/data)
## Quick Start

```bash
pip install -r requirements.txt

# Fine-tune with default settings (1 sample per category, ~11 pairs total)
python scripts/finetune_qwen3_autoqa.py \
    --vcape-r data/vcape-r-20k \
    --vcape-s data/vcape-s-20k

# Fine-tune with more samples and custom hyperparameters
python scripts/finetune_qwen3_autoqa.py \
    --vcape-r data/vcape-r-20k \
    --vcape-s data/vcape-s-20k \
    --samples-per-category 10 \
    --epochs 5 \
    --lr 1e-4

# Evaluate the fine-tuned adapter on held-out data
python scripts/infer_qwen3_autoqa.py \
    --adapter-path models/qwen3-autoqa-lora/n10 \
    --dataset vcape-r
```

## Repository Structure

```
├── scripts/                        # Core fine-tuning pipeline
│   ├── finetune_qwen3_autoqa.py    # Main entry point (sampling → formatting → training)
│   ├── finetune_sampler.py         # Stratified sampler
│   ├── training_formatter.py       # Chat-format training example generator
│   ├── fine_tuner.py               # LoRA fine-tuning with PEFT + TRL
│   └── infer_qwen3_autoqa.py      # Evaluation on held-out data
├── bedrock/                        # Batch inference & evaluation
│   ├── run_bedrock_batch_inference.py      # Cloud batch inference job submission
│   ├── evaluate_batch_outputs.py           # Metrics computation (accuracy, P/R/F1)
│   ├── autoqa_evaluator.py                 # vLLM-based local evaluation
│   ├── autoqa_evaluator_single_image.py    # Single-image variant
│   ├── autoqa_evaluator_rejection_reason.py        # With rejection reason output
│   ├── autoqa_evaluator_rejection_reason_thinking.py  # With extended thinking
│   ├── evaluate_dataset.py                 # Dataset evaluation framework
│   ├── extract_predictions.py              # Glue ETL for prediction extraction
│   ├── prepare_inference_batch_transform_payload.py  # SageMaker payload prep
│   ├── launch_batch_transform_job.py       # SageMaker batch transform launcher
│   ├── create_qwen_sagemaker_model.py      # SageMaker model registration
│   └── *.sh                                # Shell orchestration scripts
├── requirements.txt
└── README.md
```
## Batch Inference

For large-scale evaluation across multiple VLMs (Claude, Nova, Llama, Mistral), see `bedrock/`:

```bash
# Prepare and submit batch inference jobs
python bedrock/run_bedrock_batch_inference.py \
    --dataset data/vcape-r-20k \
    --s3-bucket your-s3-bucket \
    --s3-prefix batch_input/vcape-r-20k \
    --batch-size 500 \
    --submit-jobs

# Evaluate batch outputs
python bedrock/evaluate_batch_outputs.py \
    --output-dir outputs/ \
    --datasets vcape-s-20k=data/vcape-s-20k vcape-r-20k=data/vcape-r-20k
```

## VLLM Inference

For local evaluation across multiple VLMs (Gemma, Qwen):

```bash
# In one terminal set up VLLM
vllm serve "Qwen/Qwen3-VL-8B-Instruct"\
    --gpu-memory-utilization 1 \
    --max-model-len 10072 \
    --port 8765 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 128 \
    --limit-mm-per-prompt '{"image": 3, "video": 0}' \
    
# In another run evaluate dataset
python evaluate_dataset.py --evaluator autoqa --dataset ./dataset \
        --model Qwen/Qwen3-VL-8B-Instruct --vllm-port 8765 --num-threads 40

```


## Finetuning Pipeline Overview

The fine-tuning pipeline has three phases, orchestrated by a single entry-point script:

1. **Stratified Sampling** — select N image pairs per rejection-reason category (+ N accepted) from each dataset split.
2. **Training Data Formatting** — convert sampled pairs into Qwen3-VL chat-format examples with ground-truth chain-of-thought reasoning.
3. **LoRA Fine-Tuning** — train lightweight adapter weights on top of the frozen base model.


## CLI Arguments (finetune_qwen3_autoqa.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--vcape-r` | `data/vcape-r-20k` | Path to the vcape-r dataset |
| `--vcape-s` | `data/vcape-s-20k` | Path to the vcape-s dataset |
| `--samples-per-category` | `1` | Number of image pairs per rejection reason + accepted |
| `--base-model` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID for the base VLM |
| `--output-dir` | `models/qwen3-autoqa-lora` | Root dir for adapter weights |
| `--lora-rank` | `16` | Rank of LoRA decomposition matrices |
| `--lora-alpha` | `32` | LoRA scaling factor |
| `--lr` | `2e-4` | Peak learning rate |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `1` | Per-device batch size |
| `--seed` | `42` | Random seed for sampling |
| `--resample` | (flag) | Force re-sampling even if cached data exists |

## Dataset Format

Input datasets must be in HuggingFace `datasets.save_to_disk()` format with these columns:

- `xsource_image` — PIL Image of the source product in a lifestyle setting
- `xtarget_image` — PIL Image of the generated product on white background
- `label` — `"accepted"` or `"rejected"`
- `rejection_reason` — category string (e.g. `"Geometry Artifacts"`, `"wrong_orientation"`)
- `object_description` — text description of the product

## Output

Adapter weights are saved under `<output-dir>/n<N>/`:

- `adapter_config.json` — PEFT adapter configuration
- `adapter_model.safetensors` — LoRA weights
- `training_log.json` — per-step training metrics
- `metrics_summary.json` — structured metrics for plotting


