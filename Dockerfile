FROM nvcr.io/nvidia/pytorch:25.11-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Use BuildKit cache mount to speed up pip installs across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt && \
    pip3 uninstall -y pynvml 2>/dev/null || true

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/outputs /workspace/data /workspace/logs /workspace/checkpoints

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_CACHE=/root/.cache/huggingface/hub
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/hub

# Default command
CMD ["bash", "-c", "echo '=== Qwen3-32B GRPO Training with TRL + Unsloth ==='; echo ''; echo 'Available training tasks:'; echo ''; echo '1. Math Reasoning (GSM8K):'; echo '  - Run training: python train_grpo.py'; echo '  - With custom config: python train_grpo.py --config configs/custom.yaml'; echo ''; echo '2. Link Search Agent:'; echo '  - Quick test: python train_grpo_linksearch.py --mode simple'; echo '  - Full training: python train_grpo_linksearch.py --mode masked'; echo '  - With detailed logs: python train_grpo_linksearch.py --mode masked --enable-detailed-logging'; echo '  - Or use script: ./scripts/train_linksearch.sh --mode masked'; echo ''; echo 'Other commands:'; echo '  - Test setup: python test_linksearch_setup.py'; echo '  - Run evaluation: python eval_model.py --checkpoint outputs/checkpoint-xxx'; echo ''; exec bash"]
