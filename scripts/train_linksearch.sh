#!/bin/bash
# Start GRPO training for Link Search Agent

set -e

echo "🚀 Starting GRPO Training for Link Search Agent"
echo "=================================================="

# Parse arguments
MODE="masked"
RESUME=""
RESUME_BEST=""
ENABLE_LOGGING=""

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--mode masked|rollout|simple] [--resume] [--resume-best] [--enable-detailed-logging]"
            exit 1
            ;;
    esac
done

echo "Mode: $MODE"
echo "Resume: ${RESUME:-No}"
echo "Resume Best: ${RESUME_BEST:-No}"
echo "Detailed Logging: ${ENABLE_LOGGING:-Disabled}"
echo "=================================================="
echo ""

# Check required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not set. You may not be able to download datasets."
fi

if [ -z "$PROFILE_DB_PATH" ]; then
    echo "⚠️  Warning: PROFILE_DB_PATH not set. Using default path."
    export PROFILE_DB_PATH="/workspace/link_search_agent/data/profiles.db"
fi

echo "Database: $PROFILE_DB_PATH"
echo ""

# Start training
python train_grpo_linksearch.py \
    --mode "$MODE" \
    $RESUME \
    $RESUME_BEST \
    $ENABLE_LOGGING

echo ""
echo "✅ Training completed!"
echo "=================================================="
