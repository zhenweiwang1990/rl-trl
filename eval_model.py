#!/usr/bin/env python3
"""
Evaluation script for trained Qwen3-32B model

Usage:
    python eval_model.py --checkpoint outputs/qwen3-32b-grpo/final
    python eval_model.py --checkpoint outputs/qwen3-32b-grpo/checkpoint-1000
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(checkpoint_path: str):
    """Load trained model and tokenizer."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
        load_in_4bit=False,
        dtype=None,
        device_map="auto",
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    logger.info("✓ Model loaded successfully")
    return model, tokenizer


def evaluate_on_dataset(model, tokenizer, dataset_name: str = "openai/gsm8k", split: str = "test", max_samples: int = 100):
    """Evaluate model on a dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Handle different dataset formats
    if dataset_name == "openai/gsm8k":
        dataset = load_dataset(dataset_name, "main")
    elif dataset_name == "openai/summarize_from_feedback":
        dataset = load_dataset(dataset_name, "comparisons")
    else:
        dataset = load_dataset(dataset_name)
    
    if split not in dataset:
        logger.warning(f"Split '{split}' not found, using 'train'")
        split = "train"
    
    eval_data = dataset[split]
    if max_samples:
        eval_data = eval_data.select(range(min(max_samples, len(eval_data))))
    
    logger.info(f"Evaluating on {len(eval_data)} samples...")
    
    correct = 0
    total = 0
    
    for example in tqdm(eval_data, desc="Evaluating"):
        prompt = example.get("query", example.get("question", ""))
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        # Simple evaluation (customize based on your task)
        if "answer" in example:
            expected = str(example["answer"])
            if expected.lower() in response.lower():
                correct += 1
        
        total += 1
        
        if total <= 3:  # Print first 3 examples
            logger.info(f"\n--- Example {total} ---")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Response: {response[:200]}...")
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation Results:")
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Correct: {correct}")
    logger.info(f"  Accuracy: {accuracy:.2%}")
    logger.info(f"{'='*80}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to evaluate")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    
    # Evaluate
    accuracy = evaluate_on_dataset(
        model,
        tokenizer,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
