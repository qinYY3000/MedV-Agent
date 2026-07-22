"""
BioMedCLIP Classification API Server
=====================================
零样本医学图像分类，基于 Microsoft BioMedCLIP (PubMedBERT-256 + ViT-Base/16)。
不需要在 BUSI 上训练，直接对比图像与文本嵌入实现分类。

本地加载: 设置环境变量 BIOMEDCLIP_MODEL_DIR 指向本地模型目录
下载地址: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

端口: 8267
启动: bash RL-verl/api_server/run_classification_api.sh

依赖: pip install open-clip-torch==2.27.0
"""

import io
import os
import json
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, Form, HTTPException

# ------------------------------------------------------------
# BioMedCLIP 分类器 (零样本, 不需要训练)
# ------------------------------------------------------------

class BioMedCLIPClassifier:
    """BioMedCLIP 零样本医学图像分类器。
    
    原理: 计算图像嵌入与各候选文本嵌入的余弦相似度,
         取相似度最高的类别作为分类结果。
    
    本地加载方式:
      1. 在有网络的机器上下载:
         git clone https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
      2. 复制整个目录到离线机器
      3. 设置 BIOMEDCLIP_MODEL_DIR 指向该目录
    """
    
    def __init__(self, model_dir: str = None, device: str = None):
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # 确定模型路径
        if model_dir and os.path.isdir(model_dir):
            # 本地路径: 目录下应有 open_clip_config.json 和 open_clip_pytorch_model.bin
            config_path = os.path.join(model_dir, "open_clip_config.json")
            weight_path = os.path.join(model_dir, "open_clip_pytorch_model.bin")
            
            if not os.path.exists(weight_path):
                # 尝试在 snapshots 子目录中查找
                snapshots_dir = os.path.join(model_dir, "snapshots")
                if os.path.isdir(snapshots_dir):
                    for d in os.listdir(snapshots_dir):
                        p = os.path.join(snapshots_dir, d)
                        if os.path.isdir(p) and os.path.exists(os.path.join(p, "open_clip_pytorch_model.bin")):
                            weight_path = os.path.join(p, "open_clip_pytorch_model.bin")
                            config_path = os.path.join(p, "open_clip_config.json")
                            break
            
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"BioMedCLIP weight file not found in {model_dir}.\n"
                    "Expected: open_clip_pytorch_model.bin\n"
                    "Download from: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
            
            print(f"[BioMedCLIP] Loading from local: {weight_path}")
            # 离线加载: 构造 HF 缓存目录结构使 create_model_from_pretrained 认为文件已缓存
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # 1) 构造 HF 缓存目录
            import hashlib
            repo_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            cache_key = "models--" + repo_id.replace("/", "--")
            
            hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            model_cache_dir = os.path.join(hf_cache_dir, cache_key)
            snapshots_dir = os.path.join(model_cache_dir, "snapshots")
            refs_dir = os.path.join(model_cache_dir, "refs")
            blobs_dir = os.path.join(model_cache_dir, "blobs")
            
            # 2) 复制文件到 blobs 目录
            os.makedirs(blobs_dir, exist_ok=True)
            os.makedirs(refs_dir, exist_ok=True)
            os.makedirs(snapshots_dir, exist_ok=True)
            
            for fname in ["open_clip_config.json", "open_clip_pytorch_model.bin"]:
                src = os.path.join(model_dir, fname)
                if os.path.exists(src):
                    dst = os.path.join(blobs_dir, fname)
                    import shutil
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
            
            # 3) 创建 blob hash → snapshot 软链接
            commit_hash = "local"
            snapshot_dir = os.path.join(snapshots_dir, commit_hash)
            os.makedirs(snapshot_dir, exist_ok=True)
            
            for fname in ["open_clip_config.json", "open_clip_pytorch_model.bin"]:
                blob_path = os.path.join(blobs_dir, fname)
                link_path = os.path.join(snapshot_dir, fname)
                if os.path.exists(blob_path) and not os.path.exists(link_path):
                    os.symlink(blob_path, link_path)
            
            # 4) 写 refs/main
            with open(os.path.join(refs_dir, "main"), "w") as f:
                f.write(commit_hash)
            
            # 5) 设置 HF_HOME 指向 ~/.cache/huggingface
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
            
            import open_clip
            model, self.preprocess = open_clip.create_model_from_pretrained(
                f"hf-hub:{repo_id}",
                force_image_size=224,
                cache_dir=hf_cache_dir,
            )
            self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{repo_id}")
        else:
            # 尝试从 HuggingFace Hub 下载 (需要网络)
            print("[BioMedCLIP] Trying to load from HuggingFace Hub...")
            try:
                model, self.preprocess = create_model_from_pretrained(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
                self.tokenizer = get_tokenizer(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load BioMedCLIP from HuggingFace: {e}\n"
                    "Please download the model locally and set BIOMEDCLIP_MODEL_DIR.\n"
                    "Download: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                )
        
        self.model = model.to(device).eval()
        self.context_length = 256
        print(f"[BioMedCLIP] Model loaded on {device}")
    
    @torch.no_grad()
    def classify(self, image: Image.Image, text_prompts: list, labels: list = None) -> dict:
        """零样本分类图像。

        Args:
            image: PIL Image (RGB)
            text_prompts: 候选文本描述列表 (必填，由调用方根据数据集/模态传入)
                          例如 BUSI:  ["a benign breast tumor in ultrasound",
                                       "a malignant breast tumor in ultrasound",
                                       "a normal breast ultrasound"]
                          例如 Kvasir: ["a polyp in colonoscopy image",
                                        "a normal colonoscopy image"]
            labels: 对应的标签名列表 (可选，默认用 text_prompts 本身)

        Returns:
            {"label": str, "confidence": float, "all_probs": dict}
        """
        if text_prompts is None or len(text_prompts) < 2:
            raise ValueError("text_prompts must be a list with >= 2 items")

        if labels is None:
            labels = text_prompts  # 直接用 prompt 文本做 label 名
        elif len(labels) != len(text_prompts):
            raise ValueError("labels must have same length as text_prompts")

        # 预处理图像
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize 文本 (不加模板前缀，调用方自己写好完整描述)
        texts = self.tokenizer(
            text_prompts,
            context_length=self.context_length
        ).to(self.device)

        # 前向推理
        image_features, text_features, logit_scale = self.model(img_tensor, texts)

        # 计算相似度
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        probs = logits[0].cpu().numpy()

        # 获取最高概率的类别
        pred_idx = int(probs.argmax())

        all_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}

        return {
            "label": labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "all_probs": all_probs
        }
    
    @torch.no_grad()
    def classify_region(self, image: Image.Image, bbox: list, text_prompts: list = None) -> dict:
        """对图像指定区域分类 (裁剪后分类)。"""
        cropped = image.crop(tuple(bbox))
        return self.classify(cropped, text_prompts)


# ------------------------------------------------------------
# FastAPI 应用
# ------------------------------------------------------------

app = FastAPI(title="BioMedCLIP Classification API (Zero-Shot)", version="2.0")

classifier: BioMedCLIPClassifier = None


@app.on_event("startup")
async def startup():
    global classifier
    model_dir = os.environ.get("BIOMEDCLIP_MODEL_DIR", "")
    
    if not model_dir:
        # 常见位置
        candidates = [
            "RL-verl/api_server/biomedclip_model",
            str(os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")),
        ]
        for c in candidates:
            if os.path.isdir(c):
                model_dir = c
                break
    
    classifier = BioMedCLIPClassifier(model_dir=model_dir if model_dir else None)
    print("[API] BioMedCLIP Classification API ready on port 8267")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "BioMedCLIP", "device": classifier.device}


@app.post("/classify")
async def classify(
    image: UploadFile,
    text_prompts: str = Form(...),
    labels: str = Form(default=None),
):
    """零样本分类整张图像。

    必选参数:
        text_prompts: JSON 字符串, 如
          BUSI → '["a benign breast tumor in ultrasound", "a malignant breast tumor in ultrasound", "a normal breast ultrasound"]'
          Kvasir → '["a polyp in colonoscopy image", "a normal colonoscopy image"]'
          COVID → '["chest x-ray with COVID-19 pneumonia", "chest x-ray with lung opacity", "normal chest x-ray", "chest x-ray with viral pneumonia"]'

    可选参数:
        labels: JSON 字符串, 对应的标签名。不传则用 text_prompts 原文作为 label。
    """
    import time
    t0 = time.time()

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    prompts = json.loads(text_prompts)
    if not isinstance(prompts, list) or len(prompts) < 2:
        raise HTTPException(400, "text_prompts must be a JSON list with >= 2 items")

    label_list = None
    if labels:
        label_list = json.loads(labels)
        if not isinstance(label_list, list) or len(label_list) != len(prompts):
            raise HTTPException(400, "labels must be a JSON list with same length as text_prompts")

    result = classifier.classify(img, text_prompts=prompts, labels=label_list)

    t1 = time.time()
    result["time_ms"] = round((t1 - t0) * 1000, 1)
    return result


@app.post("/classify/region")
async def classify_region(
    image: UploadFile,
    bbox: str = Form(...),
    text_prompts: str = Form(...),
    labels: str = Form(default=None),
):
    """零样本分类图像指定区域。

    Args:
        bbox: JSON "[x1, y1, x2, y2]"
        text_prompts: JSON 候选描述列表
        labels: JSON 标签名列表 (可选)
    """
    import time
    t0 = time.time()

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    bbox_list = json.loads(bbox)
    if len(bbox_list) != 4:
        raise HTTPException(400, "bbox must be [x1, y1, x2, y2]")

    prompts = json.loads(text_prompts)
    if not isinstance(prompts, list) or len(prompts) < 2:
        raise HTTPException(400, "text_prompts must be a JSON list with >= 2 items")

    label_list = None
    if labels:
        label_list = json.loads(labels)

    result = classifier.classify_region(img, bbox_list, text_prompts=prompts, labels=label_list)

    t1 = time.time()
    result["time_ms"] = round((t1 - t0) * 1000, 1)
    result["region"] = bbox_list
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8267))
    uvicorn.run(app, host="0.0.0.0", port=port)
