#!/bin/bash
# Docker one-click training script for Link Search Agent
# Ensures proper .env file loading and supports resume training

set -e

echo "🐳 Docker Training Script for Link Search Agent"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
MODE="masked"
RESUME=""
RESUME_BEST=""
ENABLE_LOGGING=""
ENV_FILE=".env"
CONTAINER_NAME="qwen3-grpo-linksearch-train"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --resume-best)
            RESUME_BEST="--resume_best"
            shift
            ;;
        --enable-detailed-logging)
            ENABLE_LOGGING="--enable-detailed-logging"
            shift
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode [masked|rollout|simple]  Training mode (default: masked)"
            echo "  --resume                         Resume from last checkpoint"
            echo "  --resume-best                    Resume from best checkpoint"
            echo "  --enable-detailed-logging        Enable detailed logging"
            echo "  --env-file FILE                  Path to .env file (default: .env)"
            echo "  --container-name NAME            Container name (default: qwen3-grpo-linksearch-train)"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Error: Environment file '$ENV_FILE' not found!${NC}"
    echo ""
    echo "Please create a .env file with your configuration."
    echo "You can use env.linksearch.example as a template:"
    echo ""
    echo "  cp env.linksearch.example .env"
    echo "  nano .env  # Edit with your values"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Found environment file: $ENV_FILE"

# Validate .env file contains required variables
REQUIRED_VARS=("HF_TOKEN" "PROFILE_DB_PATH")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if ! grep -q "^${var}=" "$ENV_FILE"; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: The following required variables are missing in $ENV_FILE:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker is not installed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker is installed"

# Check if nvidia-docker/nvidia-container-runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  Warning: nvidia-docker runtime may not be properly configured${NC}"
    echo "GPU support might not work. Please ensure nvidia-container-toolkit is installed."
fi

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}⚠️  Container '$CONTAINER_NAME' already exists. Removing it...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Resume: ${RESUME:-No}"
echo "  Resume Best: ${RESUME_BEST:-No}"
echo "  Detailed Logging: ${ENABLE_LOGGING:-Disabled}"
echo "  Environment File: $ENV_FILE"
echo "  Container Name: $CONTAINER_NAME"
echo "=================================================="
echo ""

# Check if Docker image exists (built by scripts/build.sh)
IMAGE_NAME="qwen3-grpo:latest"
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}$"; then
    echo -e "${RED}❌ Error: Docker image '$IMAGE_NAME' not found!${NC}"
    echo ""
    echo "Please build the Docker image first using:"
    echo "  bash scripts/build.sh"
    echo ""
    exit 1
else
    echo -e "${GREEN}✓${NC} Docker image '$IMAGE_NAME' found"
fi

# Create necessary directories
mkdir -p outputs data logs checkpoints wandb

# Prepare training command
TRAIN_CMD="python train_grpo_linksearch.py --mode $MODE $RESUME $RESUME_BEST $ENABLE_LOGGING"

echo ""
echo "🚀 Starting Docker container..."
echo "Training command: $TRAIN_CMD"
echo ""

# Run Docker container with proper environment file mounting
docker run -d --rm \
    --restart always \
    --name "$CONTAINER_NAME" \
    --runtime=nvidia \
    --gpus all \
    --env-file "$ENV_FILE" \
    --shm-size=32g \
    -v "$PWD:/workspace" \
    -v "$PWD/outputs:/workspace/outputs" \
    -v "$PWD/data:/workspace/data" \
    -v "$PWD/logs:/workspace/logs" \
    -v "$PWD/checkpoints:/workspace/checkpoints" \
    -v "$PWD/wandb:/workspace/wandb" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -w /workspace \
    qwen3-grpo:latest \
    bash -c "
        # Source environment variables to make them available in subshells
        set -a
        source /workspace/$ENV_FILE
        set +a
        
        # Show loaded environment (for debugging)
        echo '==================================================='
        echo '📋 Loaded Environment Variables:'
        echo '==================================================='
        echo \"HF_TOKEN: \${HF_TOKEN:0:10}...\"
        echo \"PROFILE_DB_PATH: \$PROFILE_DB_PATH\"
        echo \"OUTPUT_DIR: \$OUTPUT_DIR\"
        echo \"WANDB_PROJECT: \$WANDB_PROJECT\"
        echo \"WANDB_MODE: \$WANDB_MODE\"
        echo '==================================================='
        echo ''
        
        # Verify database file exists (if path is set)
        if [ -n \"\$PROFILE_DB_PATH\" ] && [ ! -f \"\$PROFILE_DB_PATH\" ]; then
            echo \"⚠️  Warning: Database file not found at: \$PROFILE_DB_PATH\"
            echo \"Please ensure the database file is in the correct location.\"
            echo \"\"
        fi
        
        # Run training
        echo '🚀 Starting training...'
        echo ''
        $TRAIN_CMD
        
        # Training completed
        EXIT_CODE=\$?
        echo ''
        if [ \$EXIT_CODE -eq 0 ]; then
            echo '✅ Training completed successfully!'
        else
            echo '❌ Training failed with exit code:' \$EXIT_CODE
        fi
        echo '==================================================='
        
        exit \$EXIT_CODE
    "

docker logs -f "$CONTAINER_NAME"
