#!/bin/bash
# Build Docker image for Qwen3-32B GRPO training

set -e

echo "🐳 Building Docker image for Qwen3-32B GRPO training..."
echo "=================================================="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
    -t qwen3-grpo:latest \
    -f Dockerfile \
    .

echo ""
echo "✅ Docker image built successfully!"
echo "=================================================="
echo "Image: qwen3-grpo:latest"
echo ""
echo "Next steps:"
echo "  1. Run container: bash scripts/run.sh"
echo "  2. Start training: bash scripts/train.sh"
echo "=================================================="
