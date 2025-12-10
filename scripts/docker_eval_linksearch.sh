#!/bin/bash
# Docker evaluation script for Link Search Agent
# Evaluates a trained model in a Docker container

set -e

echo "🐳 Docker Evaluation Script for Link Search Agent"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CHECKPOINT=""
EVAL_SIZE="100"
NUM_ROLLOUTS="1"
SPLIT="test"
ENV_FILE=".env"
CONTAINER_NAME="qwen3-grpo-linksearch-eval"
SAVE_RESULTS=""
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --eval-size)
            EVAL_SIZE="$2"
            shift 2
            ;;
        --num-rollouts)
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --save-results)
            SAVE_RESULTS="--save-results"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo ""
            echo "Usage: $0 --checkpoint <path> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint PATH            Path to model checkpoint (e.g., outputs/grpo_linksearch_masked/final)"
            echo ""
            echo "Options:"
            echo "  --eval-size N                Number of evaluation queries (default: 100)"
            echo "  --num-rollouts N             Number of rollouts per query (default: 1)"
            echo "  --split train|test           Dataset split to evaluate on (default: test)"
            echo "  --env-file FILE              Path to .env file (default: .env)"
            echo "  --container-name NAME        Container name (default: qwen3-grpo-linksearch-eval)"
            echo "  --save-results               Save results to JSON file"
            echo "  --verbose                    Enable verbose output"
            echo "  -h, --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate final model"
            echo "  $0 --checkpoint outputs/grpo_linksearch_masked/final"
            echo ""
            echo "  # Evaluate specific checkpoint with 200 queries"
            echo "  $0 --checkpoint outputs/grpo_linksearch_masked/checkpoint-0100 --eval-size 200"
            echo ""
            echo "  # Evaluate with multiple rollouts and save results"
            echo "  $0 --checkpoint outputs/grpo_linksearch_masked/best_model --num-rollouts 3 --save-results"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}❌ Error: --checkpoint is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo -e "${RED}❌ Error: Checkpoint directory not found: $CHECKPOINT${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found checkpoint: $CHECKPOINT"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  Warning: Environment file '$ENV_FILE' not found${NC}"
    echo "Continuing without environment file..."
else
    echo -e "${GREEN}✓${NC} Found environment file: $ENV_FILE"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker is not installed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker is installed"

# Check Docker image
IMAGE_NAME="qwen3-grpo:latest"
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}$"; then
    echo -e "${RED}❌ Error: Docker image '$IMAGE_NAME' not found!${NC}"
    echo ""
    echo "Please build the Docker image first using:"
    echo "  bash scripts/build.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker image '$IMAGE_NAME' found"

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}⚠️  Container '$CONTAINER_NAME' already exists. Removing it...${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Evaluation size: $EVAL_SIZE queries"
echo "  Rollouts per query: $NUM_ROLLOUTS"
echo "  Dataset split: $SPLIT"
echo "  Save results: ${SAVE_RESULTS:-No}"
echo "  Verbose: ${VERBOSE:-No}"
echo "  Environment File: $ENV_FILE"
echo "  Container Name: $CONTAINER_NAME"
echo "=================================================="
echo ""

# Prepare evaluation command
EVAL_CMD="python eval_linksearch.py --checkpoint /workspace/$CHECKPOINT --eval-size $EVAL_SIZE --num-rollouts $NUM_ROLLOUTS --split $SPLIT $SAVE_RESULTS $VERBOSE"

echo "🚀 Starting Docker container for evaluation..."
echo "Evaluation command: $EVAL_CMD"
echo ""

# Build docker run command
DOCKER_CMD="docker run --rm \
    --name \"$CONTAINER_NAME\" \
    --runtime=nvidia \
    --gpus all"

# Add env file if it exists
if [ -f "$ENV_FILE" ]; then
    DOCKER_CMD="$DOCKER_CMD \
    --env-file \"$ENV_FILE\""
fi

# Add volumes and other options
DOCKER_CMD="$DOCKER_CMD \
    --shm-size=32g \
    -v \"$PWD:/workspace\" \
    -v \"$HOME/.cache/huggingface:/root/.cache/huggingface\" \
    -w /workspace \
    $IMAGE_NAME \
    bash -c \"
        # Source environment variables if env file exists
        if [ -f /workspace/$ENV_FILE ]; then
            set -a
            source /workspace/$ENV_FILE
            set +a
        fi
        
        # Show environment info
        echo '==================================================='
        echo '📋 Environment Info:'
        echo '==================================================='
        echo \\\"Checkpoint: $CHECKPOINT\\\"
        echo \\\"HF_TOKEN: \\\${HF_TOKEN:0:10}...\\\"
        echo \\\"PROFILE_DB_PATH: \\\$PROFILE_DB_PATH\\\"
        echo '==================================================='
        echo ''
        
        # Verify database file exists (if path is set)
        if [ -n \\\"\\\$PROFILE_DB_PATH\\\" ] && [ ! -f \\\"\\\$PROFILE_DB_PATH\\\" ]; then
            echo \\\"⚠️  Warning: Database file not found at: \\\$PROFILE_DB_PATH\\\"
            echo \\\"Please ensure the database file is in the correct location.\\\"
            echo \\\"\\\"
        fi
        
        # Run evaluation
        echo '🔍 Starting evaluation...'
        echo ''
        $EVAL_CMD
        
        # Evaluation completed
        EXIT_CODE=\\\$?
        echo ''
        if [ \\\$EXIT_CODE -eq 0 ]; then
            echo '✅ Evaluation completed successfully!'
        else
            echo '❌ Evaluation failed with exit code:' \\\$EXIT_CODE
        fi
        echo '==================================================='
        
        exit \\\$EXIT_CODE
    \""

# Execute the docker command
eval "$DOCKER_CMD"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Docker evaluation completed successfully!${NC}"
    
    # Check if results were saved
    if [ -n "$SAVE_RESULTS" ]; then
        CHECKPOINT_NAME=$(basename "$CHECKPOINT")
        RESULT_FILE="$(dirname "$CHECKPOINT")/eval_results_${CHECKPOINT_NAME}.json"
        if [ -f "$RESULT_FILE" ]; then
            echo -e "${GREEN}📁 Results saved to: $RESULT_FILE${NC}"
        fi
    fi
else
    echo -e "${RED}❌ Docker evaluation failed with exit code: $EXIT_CODE${NC}"
fi
echo "=================================================="
echo ""

exit $EXIT_CODE
