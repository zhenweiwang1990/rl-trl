.PHONY: help build run train eval test clean

help:
	@echo "Qwen3-32B GRPO Training Makefile"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make build     - Build Docker image"
	@echo "  make run       - Run Docker container"
	@echo "  make train     - Start training (default config)"
	@echo "  make eval      - Evaluate trained model"
	@echo "  make test      - Interactive testing"
	@echo "  make clean     - Clean outputs and logs"
	@echo "  make help      - Show this help message"
	@echo ""

build:
	@echo "🐳 Building Docker image..."
	bash scripts/build.sh

run:
	@echo "🚀 Running Docker container..."
	bash scripts/run.sh

train:
	@echo "📚 Starting training..."
	bash scripts/train.sh

eval:
	@echo "📊 Evaluating model..."
	python eval_model.py --checkpoint outputs/qwen3-32b-grpo/final

test:
	@echo "💬 Starting interactive test..."
	python interactive_test.py --checkpoint outputs/qwen3-32b-grpo/final

clean:
	@echo "🧹 Cleaning outputs and logs..."
	rm -rf outputs/* logs/* wandb/*
	@echo "✓ Cleaned"
