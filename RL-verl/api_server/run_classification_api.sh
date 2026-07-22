#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# === BioMedCLIP 本地模型配置 ===
# 下载方式: git clone https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# 然后设置此路径:
BIOMEDCLIP_MODEL_DIR=${BIOMEDCLIP_MODEL_DIR:-"/mnt/workspace/MedSAM-Agent/RL-verl/api_server/biomedclip_model"}
PORT=${PORT:-8267}

echo "Starting BioMedCLIP Classification API on port $PORT"
echo "Model dir: $BIOMEDCLIP_MODEL_DIR"

export BIOMEDCLIP_MODEL_DIR
export PORT

python3 "$SCRIPT_DIR/classification_api.py"
