#!/usr/bin/env python3
"""
Interactive testing script for Qwen3-32B model

Usage:
    python interactive_test.py --checkpoint outputs/qwen3-32b-grpo/final
"""

import argparse
import logging
import sys

import torch
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


def chat_loop(model, tokenizer, max_new_tokens: int = 512, temperature: float = 0.7):
    """Interactive chat loop."""
    print("\n" + "="*80)
    print("Interactive Chat with Qwen3-32B")
    print("="*80)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("="*80 + "\n")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == "clear":
                conversation_history = []
                print("\n🗑️  Conversation cleared")
                continue
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Prepare input
            input_text = tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            
            print(response)
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with trained model")
    parser.add_argument("--checkpoint", type=str, default="outputs/qwen3-32b-grpo/final", 
                        help="Path to model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)
    
    # Start chat loop
    chat_loop(model, tokenizer, args.max_new_tokens, args.temperature)


if __name__ == "__main__":
    main()
