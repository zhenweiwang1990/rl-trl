#!/bin/bash
# Run Docker container for Qwen3-32B GRPO training

set -e

echo "🚀 Starting Docker container for Qwen3-32B GRPO training..."
echo "=================================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Container name
CONTAINER_NAME="qwen3-grpo-train"

# Check if container already exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️  Container '$CONTAINER_NAME' already exists"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rm -f $CONTAINER_NAME
    else
        echo "Exiting..."
        exit 1
    fi
fi

# Check if database path is set for Link Search
DB_MOUNT=""
if [ -n "$PROFILE_DB_PATH" ] && [ -f "$PROFILE_DB_PATH" ]; then
    echo "📦 Mounting profile database: $PROFILE_DB_PATH"
    DB_MOUNT="-v $PROFILE_DB_PATH:/workspace/link_search_agent/data/profiles.db:ro"
fi

# Run container with GPU support
docker run -it --rm \
    --gpus all \
    --name $CONTAINER_NAME \
    --shm-size=32g \
    -v "$PROJECT_DIR":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$PROJECT_DIR/outputs":/workspace/outputs \
    -v "$PROJECT_DIR/data":/workspace/data \
    -v "$PROJECT_DIR/logs":/workspace/logs \
    $DB_MOUNT \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e PROFILE_DB_PATH="/workspace/link_search_agent/data/profiles.db" \
    -p 6006:6006 \
    qwen3-grpo:latest

echo ""
echo "✅ Container stopped"
echo "=================================================="
