#!/bin/bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 -m pip install "yapf,scikit-image,supervision,pycocotools,addict" -q
# === Grounding DINO 本地模型配置 ===
# 下载方式:
#   1. git clone https://github.com/IDEA-Research/GroundingDINO.git (放在 SCRIPT_DIR 下)
#   2. wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
#   3. 配置文件和权重文件放在 SCRIPT_DIR 下
# 然后设置:
GROUNDING_DINO_HOME=${GROUNDING_DINO_HOME:-"$SCRIPT_DIR/GroundingDINO"}
GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-"$SCRIPT_DIR/GroundingDINO_SwinT_OGC.py"}
GROUNDING_DINO_CHECKPOINT=${GROUNDING_DINO_CHECKPOINT:-"$SCRIPT_DIR/groundingdino_swint_ogc.pth"}
PORT=${PORT:-8266}

echo "Starting Grounding DINO Detection API on port $PORT"
echo "GD Home:   $GROUNDING_DINO_HOME"
echo "Config:    $GROUNDING_DINO_CONFIG"
echo "Checkpoint: $GROUNDING_DINO_CHECKPOINT"

export GROUNDING_DINO_HOME
export GROUNDING_DINO_CONFIG
export GROUNDING_DINO_CHECKPOINT
export PORT

# bert-base-uncased 本地路径 (离线加载, 避免 HuggingFace 下载超时)
export BERT_MODEL_PATH="${BERT_MODEL_PATH:-$SCRIPT_DIR/bert-base-uncased}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 "$SCRIPT_DIR/detection_api.py"
