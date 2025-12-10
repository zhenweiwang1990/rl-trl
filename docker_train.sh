#!/bin/bash
# Quick start script for Docker training
# This is a simplified wrapper around scripts/docker_train_linksearch.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if .env exists, if not, create from example
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "⚠️  .env file not found!"
    if [ -f "$SCRIPT_DIR/env.linksearch.example" ]; then
        echo "Creating .env from env.linksearch.example..."
        cp "$SCRIPT_DIR/env.linksearch.example" "$SCRIPT_DIR/.env"
        echo ""
        echo "✅ Created .env file. Please edit it with your configuration:"
        echo "   nano .env"
        echo ""
        echo "Required fields to update:"
        echo "  - HF_TOKEN: Your Hugging Face token"
        echo "  - PROFILE_DB_PATH: Path to your profiles database"
        echo "  - WANDB_API_KEY: (optional) Your W&B API key"
        echo ""
        exit 1
    else
        echo "❌ env.linksearch.example not found!"
        exit 1
    fi
fi

# Run the main Docker training script with all arguments passed through
exec "$SCRIPT_DIR/scripts/docker_train_linksearch.sh" "$@"
