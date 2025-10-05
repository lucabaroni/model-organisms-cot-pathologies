#!/bin/bash

# Script to push counts_kl checkpoints to Hugging Face Hub
# Usage: ./push_to_huggingface.sh [REPO_NAME] [--private] [--max-checkpoints N]

set -e  # Exit on any error

# Default values
REPO_NAME="model-organisms-cot-pathologies/counts-kl-checkpoints"
CHECKPOINTS_DIR="/root/model-organisms-cot-pathologies/counts_kl"
PRIVATE=false
MAX_CHECKPOINTS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --private)
            PRIVATE=true
            shift
            ;;
        --max-checkpoints)
            MAX_CHECKPOINTS="$2"
            shift 2
            ;;
        *)
            if [[ -z "$REPO_NAME" || "$REPO_NAME" == "model-organisms-cot-pathologies/counts-kl-checkpoints" ]]; then
                REPO_NAME="$1"
            fi
            shift
            ;;
    esac
done

echo "============================================================"
echo "Hugging Face Checkpoints Upload Script"
echo "============================================================"
echo "Repository: $REPO_NAME"
echo "Checkpoints directory: $CHECKPOINTS_DIR"
echo "Private repository: $PRIVATE"
if [[ -n "$MAX_CHECKPOINTS" ]]; then
    echo "Max checkpoints: $MAX_CHECKPOINTS"
fi
echo "============================================================"

# Check if checkpoints directory exists
if [[ ! -d "$CHECKPOINTS_DIR" ]]; then
    echo "Error: Checkpoints directory not found: $CHECKPOINTS_DIR"
    exit 1
fi

# Install dependencies
echo "Installing required dependencies..."
pip install huggingface_hub[cli] || echo "huggingface_hub may already be installed"

# Install git-lfs if not available
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    apt-get update && apt-get install -y git-lfs
    git lfs install
fi

# Authenticate with Hugging Face
echo "Setting up Hugging Face authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please authenticate with Hugging Face:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'write' permissions"
    echo "3. Run: huggingface-cli login"
    echo "4. Enter your token when prompted"
    huggingface-cli login
else
    echo "Already authenticated as: $(huggingface-cli whoami)"
fi

# Create repository
echo "Creating repository: $REPO_NAME"
if [[ "$PRIVATE" == "true" ]]; then
    huggingface-cli repo create "$REPO_NAME" --type model --private || echo "Repository may already exist"
else
    huggingface-cli repo create "$REPO_NAME" --type model --public || echo "Repository may already exist"
fi

# Upload README if it exists
if [[ -f "$CHECKPOINTS_DIR/README.md" ]]; then
    echo "Uploading README.md..."
    huggingface-cli upload "$REPO_NAME" "$CHECKPOINTS_DIR/README.md" README.md --repo-type model
fi

# Get list of checkpoint directories
echo "Finding checkpoint directories..."
CHECKPOINT_DIRS=($(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V))

if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
    echo "No checkpoint directories found!"
    exit 1
fi

echo "Found ${#CHECKPOINT_DIRS[@]} checkpoint directories"

# Limit checkpoints if specified
if [[ -n "$MAX_CHECKPOINTS" ]]; then
    CHECKPOINT_DIRS=("${CHECKPOINT_DIRS[@]:0:$MAX_CHECKPOINTS}")
    echo "Limiting to first $MAX_CHECKPOINTS checkpoints"
fi

# Upload each checkpoint
for i in "${!CHECKPOINT_DIRS[@]}"; do
    checkpoint_dir="${CHECKPOINT_DIRS[$i]}"
    checkpoint_name=$(basename "$checkpoint_dir")
    echo ""
    echo "[$((i+1))/${#CHECKPOINT_DIRS[@]}] Uploading $checkpoint_name..."
    
    if huggingface-cli upload "$REPO_NAME" "$checkpoint_dir" "$checkpoint_name" --repo-type model; then
        echo "✓ Successfully uploaded $checkpoint_name"
    else
        echo "✗ Failed to upload $checkpoint_name"
    fi
done

echo ""
echo "============================================================"
echo "Upload completed!"
echo "Repository: https://huggingface.co/$REPO_NAME"
echo "============================================================"
