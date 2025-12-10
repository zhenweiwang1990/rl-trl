#!/usr/bin/env python3
"""
GRPO Training for Qwen3-32B using TRL + Unsloth

This script trains Qwen3-32B using Group Relative Policy Optimization (GRPO)
with the TRL library and Unsloth for efficient training.

Usage:
    python train_grpo.py
    python train_grpo.py --config configs/custom.yaml
    python train_grpo.py --model unsloth/Qwen3-32B --load_in_4bit
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
from pathlib import Path

import torch
import yaml
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file or use defaults."""
    default_config = {
        "model_name": "unsloth/Qwen3-32B",
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "dataset_name": "trl-lib/gsm8k-grpo",
        "output_dir": "outputs/qwen3-32b-grpo",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "save_total_limit": 100,
        "num_generations": 4,
        "max_prompt_length": 1024,
        "max_completion_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "beta": 0.01,
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "max_grad_norm": 1.0,
        "report_to": "wandb",
        "wandb_project": "qwen3-32b-grpo",
        "wandb_name": None,
    }
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        default_config.update(user_config)
    
    return default_config


def patch_qwen3_gradient_checkpointing(model):
    """Patch Qwen3 decoder layers for gradient checkpointing compatibility."""
    import torch.utils.checkpoint
    
    checkpoint_fn = torch.utils.checkpoint.checkpoint
    candidate_layer_lists = []
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidate_layer_lists.append(model.model.layers)
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "layers")
    ):
        candidate_layer_lists.append(model.base_model.model.layers)
    
    patched = 0
    for layers in candidate_layer_lists:
        if not layers:
            continue
        for layer in layers:
            if not hasattr(layer, "_gradient_checkpointing_func"):
                layer._gradient_checkpointing_func = checkpoint_fn
                patched += 1
        if patched:
            break
    
    if patched:
        logger.info(f"✓ Patched {patched} decoder layers for gradient checkpointing")
    
    return patched


def prepare_model_and_tokenizer(config: Dict):
    """Load and prepare model and tokenizer with LoRA."""
    logger.info(f"Loading model: {config['model_name']}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,  # Auto-detect
        device_map="auto",
    )
    
    # Apply LoRA
    logger.info("Applying LoRA configuration")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
        max_seq_length=config["max_seq_length"],
    )
    
    # Patch for gradient checkpointing
    patch_qwen3_gradient_checkpointing(model)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    logger.info("✓ Model and tokenizer loaded successfully")
    return model, tokenizer


def load_and_prepare_dataset(config: Dict):
    """Load and prepare the GRPO dataset from TRL."""
    logger.info(f"Loading dataset: {config['dataset_name']}")
    
    # Load the dataset
    dataset_name = config["dataset_name"]
    
    # Handle different dataset formats
    if dataset_name == "openai/gsm8k":
        dataset = load_dataset(dataset_name, "main")
        # Convert GSM8K format to GRPO format
        def format_gsm8k(example):
            return {"prompt": example["question"]}
        dataset = dataset.map(format_gsm8k, remove_columns=dataset["train"].column_names)
    elif dataset_name == "openai/summarize_from_feedback":
        dataset = load_dataset(dataset_name, "comparisons")
        # Convert to GRPO format
        def format_summarize(example):
            return {"prompt": example.get("info", {}).get("post", "")}
        dataset = dataset.map(format_summarize)
    else:
        dataset = load_dataset(dataset_name)
        # Assume it already has "prompt" field, or try to convert
        if "question" in dataset["train"].column_names and "prompt" not in dataset["train"].column_names:
            def add_prompt(example):
                return {"prompt": example["question"]}
            dataset = dataset.map(add_prompt)
    
    logger.info(f"Dataset loaded and formatted: {dataset}")
    logger.info(f"  Train samples: {len(dataset['train'])}")
    if "test" in dataset:
        logger.info(f"  Test samples: {len(dataset['test'])}")
    elif "validation" in dataset:
        logger.info(f"  Validation samples: {len(dataset['validation'])}")
    
    # Show a sample
    logger.info(f"  Sample prompt: {dataset['train'][0]['prompt'][:100]}...")
    
    return dataset


def reward_function(samples: List[str], prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
    """
    Reward function for GRPO training.
    
    This is a simple example that rewards correct answers in math problems.
    You should customize this based on your task.
    """
    rewards = []
    
    for prompt, output in zip(prompts, outputs):
        # Extract the answer from the output
        try:
            # For GSM8K math problems
            if "####" in output:
                # Standard GSM8K answer format
                reward = 1.0
            elif any(char.isdigit() for char in output):
                # Contains numbers - partial credit
                # Reward based on output quality
                if len(output) > 100:  # Detailed explanation
                    reward = 0.8
                elif len(output) > 20:  # Some reasoning
                    reward = 0.5
                else:
                    reward = 0.3
            else:
                # No numerical content
                reward = 0.0
            
            # Penalty for very short outputs
            if len(output) < 10:
                reward *= 0.5
                
        except:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Qwen3-32B")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config["model_name"] = args.model
    if args.load_in_4bit:
        config["load_in_4bit"] = True
    if args.no_wandb:
        config["report_to"] = "none"
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Save config
    config_save_path = os.path.join(config["output_dir"], "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to {config_save_path}")
    
    # Initialize wandb
    if config["report_to"] == "wandb":
        wandb_name = config.get("wandb_name") or f"qwen3-grpo-{config['seed']}"
        wandb.init(
            project=config["wandb_project"],
            name=wandb_name,
            config=config,
        )
        logger.info(f"✓ Wandb initialized: {config['wandb_project']}/{wandb_name}")
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("GRPO Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Output: {config['output_dir']}")
    logger.info(f"Dataset: {config['dataset_name']}")
    logger.info(f"Batch size: {config['per_device_train_batch_size']}")
    logger.info(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Epochs: {config['num_train_epochs']}")
    logger.info("=" * 80)
    
    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config)
    
    # Load dataset
    dataset = load_and_prepare_dataset(config)
    
    # Configure training arguments
    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        optim=config["optim"],
        max_grad_norm=config["max_grad_norm"],
        report_to=config["report_to"],
        # GRPO specific
        num_generations=config["num_generations"],
        max_prompt_length=config.get("max_prompt_length", 1024),
        max_completion_length=config.get("max_completion_length", 512),
        temperature=config["temperature"],
        top_p=config.get("top_p", 0.9),
        beta=config["beta"],
    )
    
    # Initialize trainer
    logger.info("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )
    
    # Set UNSLOTH flag for logits
    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    
    # Start training
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(config["output_dir"], "final")
    logger.info(f"Saving final model to {final_output_dir}")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    logger.info("=" * 80)
    logger.info("✅ Training completed successfully!")
    logger.info(f"Model saved to: {final_output_dir}")
    logger.info("=" * 80)
    
    if config["report_to"] == "wandb":
        wandb.finish()


if __name__ == "__main__":
    main()
