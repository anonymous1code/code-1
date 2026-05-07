from datasets import Dataset
from colorama import Fore, init as __init_colorama
__init_colorama(autoreset=True)
from functools import partial
import re
from PIL import Image
import base64
from io import BytesIO
import os
import json
from rich import print_json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import argparse
from datetime import datetime

import requests


def extract_answer_from_reasoning(output) -> str:
    match = re.search(r'</reasoning>\s*(.*)', output, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().lower().replace("'", "")
    else:
        # Fallback - the model is unable to respect the structure
        return 'no'

def make_conversation(sample):
    SYSTEM_PROMPT = (
            "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
            "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
            )

    PROMPT = (
        "\n**Task**: Determine if the generated object is CONSISTENT with the object in the source image.\n\n"

        "**Context**:\n"
        "- The SOURCE IMAGE shows the object in a lifestyle setting (e.g., home environment, room context, in-use scenario).\n"
        "- The ADDITIONAL SOURCE IMAGE shows the same object in a different setting.\n"
        "- The GENERATED image shows the same object isolated on a white background.\n\n"

        "Consistency is ensured when the material, texture, geometry, number of element of the generated object are respected, i.e., the generated object is exactly the same as the one in the source image. "

        "**Consistency Criteria** - The following attributes MUST match:\n"
        "1. Core structural elements: Shape, form, and key geometric features\n"
        "2. Material and texture: Fabric type, wood, metal, leather, etc.\n"
        "3. Color: Primary colors and color distribution\n"
        "4. Orientation: The generated object must be oriented to the left. If the sofa is a corner sofa, the original orientation should be preserved. \n"
        "5. Quantity of essential structural components: Number of seats and cushions (for sofas).\n\n"
        
        "These objects are always allowed to be present in the generated image:\n"
        "- Decorative throw pillows on sofas/couches (unless they appear to be fixed/attached cushions)\n"
        "- Decorative accessories (blankets, throws)\n"
        "- Plaids\n"
        "- Items inside sofa pockets or compartments (e.g., magazines, papers, electronics)\n"
        "- Items inside cupholders (e.g., glasses, bottles)\n"

        "These objects are CONDITIONALLY allowed (ONLY if present in the original image):\n"
        "- Ottomans\n"

        "The following must never appear in the generated image:\n"
        "- People, pets, or body parts\n"
        "- Multiple main objects (more than one sofa)\n"
        "- Lifestyle or cluttered environments\n"
        "- Infographics, text, logos, labels, or watermarks\n"
        "- Electronics (phones, tablets, remotes, etc.), unless placed inside sofa pockets or compartments\n"
        "- Objects unrelated to the sofa that were not present in the original image\n"

        "Common rejection reasons:\n"
        "1. Geometry: Shape of the sofa, seats are missing, dimension of the chaise, etc\n"
        "2. Sofa side: Artificats introduced in the side of the sofa that are not present in the original sofa. For example: new pocket, back cushion is different, etc.\n"
        "3. Hallucinations: Some sofa parts  in the generated image are hallucinated due to occlusions in the original images. \n"
        "4. Multiple Sofas: The generated image contains multiple sofas. \n"

        "**Output Format**:\n"
        "You must reply using the reasoning and answer tag: in the reasoning provide the reasoning, in the answer the response."
        "<reasoning> ...your reasoning... </reasoning>'Yes' / 'No'"
    )

    def pil_to_base64(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode()

    # Half resolution using thumbnail (modifies in-place, maintains aspect ratio)
    for key in ['source_image_main', 'source_image_additional', 'generated_image']:
        img = sample[key].convert('RGB')
        img.thumbnail((img.size[0] // 2, img.size[1] // 2), Image.LANCZOS)
        sample[key] = img

        # img.save(f"{key}.jpg")
    
    message = [
        {"role": "system", "content":  SYSTEM_PROMPT},
        {
         "role": "user", 
         "content": [
                #### Source Image
                {
                    "type": "text",
                    "text": "\n**SOURCE IMAGE**",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_to_base64(sample['source_image_main'])}"
                    },
                },
                #### Additional Source Image
                {
                    "type": "text",
                    "text": "\n**ADDITIONAL SOURCE IMAGE**",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_to_base64(sample['source_image_additional'])}"
                    },
                },
                ## Generated Image
                {
                    "type": "text",
                    "text": "\n**GENERATED IMAGE**",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_to_base64(sample['generated_image'])}"
                    },
                },
                ### Final Prompt
                {
                    "type": "text",
                    "text": PROMPT,
                },
        ]
        },
    ]
    return message


def call_vllm_api(messages, port=8000, model_name=None,
                  temperature=0.8, max_tokens=1024, top_p=0.8):
    """Make HTTP call to vLLM server"""
    
    api_url = f"http://localhost:{port}/v1/chat/completions"
    
    payload = {
        "model": model_name ,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        # "frequency_penalty": 0.8,
        "presence_penalty": 1.5,
    }

    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"API call failed: {e}")
        print(Fore.RED + f"Response: {response.text}")
        raise e

def process_sample(sample, port=8000, model_name=None, verbose=False):
    """Process a single sample using vLLM API"""

    messages = make_conversation(sample)

    llm_generated_prompt = call_vllm_api(
        messages=messages,
        port=port,
        model_name=model_name,
    )
    if verbose:
        print(Fore.YELLOW + "[INFO] Response ->" +llm_generated_prompt)
        print(Fore.CYAN + f"[INFO] GT: {sample['is_generation_successful']}")

    answer = extract_answer_from_reasoning(llm_generated_prompt)

    return {
        **sample,
        "model_response": llm_generated_prompt,
        "model_prediction_is_generation_successfull": answer.lower().strip() == 'yes'
    }

def process_batch(batch, port=8000, model_name=None, verbose=False):
    """Process a batch of samples"""
    samples = [
        {key: batch[key][i] for key in batch.keys()}
        for i in range(len(batch[list(batch.keys())[0]]))
    ]
    
    processed_samples = [
        process_sample(sample, port, model_name, verbose) 
        for sample in samples
    ]
    
    batch_output = {
        key: [sample[key] for sample in processed_samples]
        for key in processed_samples[0].keys()
    }

    return batch_output

def calculate_metrics(dataset):
    """Calculate evaluation metrics"""
    predictions = dataset['model_prediction_is_generation_successfull']
    ground_truth = dataset['is_generation_successful']
    
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='binary', zero_division=0
    )
    
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(predictions),
        'positive_samples': int(sum(ground_truth)),
        'negative_samples': int(len(ground_truth) - sum(ground_truth)),
    }
    
    return metrics

def save_results(dataset, metrics, model_name, output_base_dir="evaluation_results", dataset_name=None):
    """Save dataset and metrics to disk"""
    model_name = model_name.replace("/", "_").replace("\\", "_")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_with_metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "dataset_name": dataset_name,
        "evaluation_date": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    print(f"{Fore.GREEN}Metrics saved to: {metrics_path}")
    
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"METRICS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TP: {metrics['true_positives']}, TN: {metrics['true_negatives']}\n")
        f.write(f"  FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}\n")
    print(f"{Fore.GREEN}Summary saved to: {summary_path}")

    reasoning_path = os.path.join(output_dir, "reasoning")
    os.makedirs(reasoning_path, exist_ok=True)
    for ix, sample in enumerate(dataset):
        reasoning = sample['model_response']

        with open(os.path.join(reasoning_path, f"{ix:09d}.txt"), 'w') as f:
            f.write(reasoning)

    
    return output_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate the AutoQA dataset using a VLM")
   

    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Base model name")
    
    parser.add_argument("--output-dir", type=str, default="auto_qa_sofa_v0_evaluation",
                        help="Output directory for results")
    
    parser.add_argument("--dataset", type=str, default="autoqa_dataset",
                        help="Dataset")
    
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with limited samples")
    
    parser.add_argument("--vllm-port", type=int, default=8765,
                        help="Port for vLLM server")
    
    parser.add_argument("--num-proc", type=int, default=40,
                        help="Number of process in parallel")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Port for vLLM server")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    BATCH_SIZE = 1

    # Load dataset
    dataset = Dataset.load_from_disk(args.dataset)
    # dataset = dataset['test']
    dataset = dataset.shuffle(seed=42)

    if args.debug:
        dataset = dataset.select(range(1))
    print(Fore.GREEN + f"Dataset '{args.dataset}' Loaded with {len(dataset)} samples")

    
    print(Fore.CYAN + f"\n{'=' * 60}")
    print(Fore.CYAN + f"Starting Evaluation:")
    print(Fore.CYAN + f"{'=' * 60}\n")
    
    # Process the dataset
    func_process_sample = partial(
        process_batch,
        port=args.vllm_port,
        model_name=args.model_path,
        verbose=args.debug
    )
    
    updated_dataset = dataset.map(
        function=func_process_sample,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=dataset.column_names,
        desc="Processing samples",
        num_proc=args.num_proc,
        load_from_cache_file=False,
    )

    # Calculate metrics
    print(f"\n{Fore.YELLOW}Calculating metrics...")
    metrics = calculate_metrics(updated_dataset)
    
    # Print metrics
    print_json(json.dumps(metrics))
    
    # Save results
    output_dir = save_results(
        updated_dataset,
        metrics,
        args.model_path,
        args.output_dir,
        dataset_name=args.dataset
    )
    
    print(f"\n{Fore.GREEN}✓ Evaluation complete!")
    print(f"{Fore.GREEN}✓ All results saved to: {args.output_dir}")


    print(Fore.GREEN + "✓ Cleanup complete")