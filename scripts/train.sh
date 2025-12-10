#!/bin/bash
# Start GRPO training for Qwen3-32B

set -e

echo "🚀 Starting GRPO Training for Qwen3-32B"
echo "=================================================="

# Parse arguments
CONFIG="configs/default.yaml"
RESUME=""
NO_WANDB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --no-wandb)
            NO_WANDB="--no_wandb"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration: $CONFIG"
echo "Resume: ${RESUME:-No}"
echo "Wandb: ${NO_WANDB:-Enabled}"
echo "=================================================="
echo ""

# Start training
python train_grpo.py \
    --config "$CONFIG" \
    $RESUME \
    $NO_WANDB

echo ""
echo "✅ Training completed!"
echo "=================================================="
