"""
Grounding DINO Detection API Server
====================================
零样本开放词汇目标检测，基于 IDEA-Research Grounding DINO。
不需要在 BUSI 上训练——给定文本描述 (如 "breast tumor")，直接检测。

本地加载: 设置 GROUNDING_DINO_MODEL_DIR 指向本地模型目录
          (需包含 config.py + 权重文件)

T4 配置 (推荐): GroundingDINO_SwinT_OGC.py + groundingdino_swint_ogc.pth
高性能配置:       GroundingDINO_SwinB.cfg.py + groundingdino_swinb_cogcoor.pth

端口: 8266
启动: bash RL-verl/api_server/run_detection_api.sh

依赖:
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git
    或 clone 到本地后 pip install -e .
"""

import io
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.ops
from fastapi import FastAPI, UploadFile, Form, HTTPException
from pathlib import Path

# ------------------------------------------------------------
# Grounding DINO 检测器 (零样本, 不需要训练)
# ------------------------------------------------------------

class GroundingDINODetector:
    """Grounding DINO 零样本检测器。
    
    原理: 将图像和文本描述输入 Grounding DINO,
         直接输出与文本匹配的检测框，无需在目标数据集上训练。
    
    本地加载方式:
      1. 在有网络的机器上下载:
         git clone https://github.com/IDEA-Research/GroundingDINO.git
         wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
      2. 复制到离线机器
      3. 设置 GROUNDING_DINO_HOME 指向 GroundingDINO 目录
         GROUNDING_DINO_CONFIG 和 GROUNDING_DINO_CHECKPOINT 指向配置文件/权重
    """
    
    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        grounding_dino_home: str = None,
        device: str = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # 确定 GroundingDINO 目录
        if grounding_dino_home:
            gd_home = Path(grounding_dino_home)
        else:
            candidates = [
                Path(__file__).parent / "GroundingDINO",
                Path("third_party/GroundingDINO"),
                Path.home() / "GroundingDINO",
            ]
            gd_home = next((c for c in candidates if c.is_dir()), None)
        
        if gd_home and str(gd_home) not in sys.path:
            sys.path.insert(0, str(gd_home))
        
        # 确定配置文件 (放置在 api_server 目录下)
        if config_path and os.path.exists(config_path):
            pass
        else:
            # 使用 T4 适配配置 (Swin-Tiny)
            local_config = Path(__file__).parent / "GroundingDINO_SwinT_OGC.py"
            if local_config.exists():
                config_path = str(local_config)
            elif gd_home:
                config_path = str(gd_home / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
            else:
                raise FileNotFoundError(
                    "Cannot find GroundingDINO config. "
                    "Set GROUNDING_DINO_CONFIG or clone GroundingDINO repo."
                )
        
        # 确定权重文件
        if checkpoint_path and os.path.exists(checkpoint_path):
            pass
        else:
            candidates = [
                Path(__file__).parent / "groundingdino_swint_ogc.pth",
                gd_home / "weights" / "groundingdino_swint_ogc.pth" if gd_home else None,
            ]
            checkpoint_path = next(
                (str(c) for c in candidates if c and c.exists()), None
            )
            if not checkpoint_path:
                raise FileNotFoundError(
                    "Cannot find GroundingDINO checkpoint (groundingdino_swint_ogc.pth).\n"
                    "Download: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth\n"
                    "Place it in: RL-verl/api_server/"
                )
        
        print(f"[GroundingDINO] Config: {config_path}")
        print(f"[GroundingDINO] Checkpoint: {checkpoint_path}")
        print(f"[GroundingDINO] GD Home: {gd_home}")
        
        # 加载模型
        from groundingdino.util.inference import Model
        self.model_instance = Model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=device
        )
        print(f"[GroundingDINO] Model loaded on {device}")
    
    def detect(self, image: Image.Image, text_prompt: str) -> dict:
        """零样本检测。
        
        Args:
            image: PIL Image (RGB)
            text_prompt: 目标描述, 如 "breast tumor ." (注意: GroundingDINO 需要末尾加句号)
        
        Returns:
            {"boxes": [{"bbox": [x1,y1,x2,y2], "score": float, "label": str}, ...],
             "num_detections": int}
        """
        # Grounding DINO 要求文本以 . 结束
        if not text_prompt.strip().endswith("."):
            text_prompt = text_prompt.strip() + " ."
        
        # 推理
        boxes, scores, phrases = self.model_instance.predict_with_caption(
            image=image,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        
        # 格式化结果
        result_boxes = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            result_boxes.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": round(float(scores[i]), 4),
                "label": phrases[i] if i < len(phrases) else ""
            })
        
        return {
            "boxes": result_boxes,
            "num_detections": len(result_boxes)
        }
    
    def draw_boxes(self, image: Image.Image, boxes: list) -> Image.Image:
        """在原图上绘制检测框 (蓝色矩形)。"""
        if image.mode != "RGB":
            img = image.convert("RGB")
        else:
            img = image.copy()
        
        draw = ImageDraw.Draw(img)
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            label = f"{box.get('label', 'obj')} {box['score']:.2f}"
            draw.text((x1 + 2, y1 - 15), label, fill="blue")
        
        return img


# ------------------------------------------------------------
# 会话管理
# ------------------------------------------------------------

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create(self, sid: str, image: Image.Image):
        self.sessions[sid] = {"image": image, "image_size": image.size}
    
    def get(self, sid: str):
        s = self.sessions.get(sid)
        if s is None:
            raise HTTPException(404, f"Session {sid} not found")
        return s
    
    def delete(self, sid: str):
        self.sessions.pop(sid, None)


# ------------------------------------------------------------
# FastAPI 应用
# ------------------------------------------------------------

app = FastAPI(title="Grounding DINO Detection API (Zero-Shot)", version="2.0")

detector = None
sessions = SessionManager()


@app.on_event("startup")
async def startup():
    global detector
    detector = GroundingDINODetector(
        config_path=os.environ.get("GROUNDING_DINO_CONFIG"),
        checkpoint_path=os.environ.get("GROUNDING_DINO_CHECKPOINT"),
        grounding_dino_home=os.environ.get("GROUNDING_DINO_HOME"),
    )
    print("[API] Grounding DINO Detection API ready on port 8266")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "GroundingDINO",
        "device": detector.device,
        "box_threshold": detector.box_threshold,
        "text_threshold": detector.text_threshold
    }


@app.post("/detect")
async def detect(
    image: UploadFile,
    caption: str = Form(...),
    box_threshold: float = Form(default=0.35),
    text_threshold: float = Form(default=0.25),
):
    """直接检测（无会话模式）。

    必选参数:
        caption: 目标描述文本, 如
          BUSI → "breast tumor ." (注意: GDINO 建议末尾加句号)
          Kvasir → "polyp ."
          COVID → "lung opacity ."
          TN3K → "thyroid nodule ."
          AbdomenCT/MR → "liver . kidney . spleen ." (多个目标用 . 分隔)
    """
    import time
    t0 = time.time()
    
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    result = detector.detect(img, text_prompt=caption)
    
    t1 = time.time()
    result["time_ms"] = round((t1 - t0) * 1000, 1)
    return result


@app.post("/detect/session/create")
async def create_session(image: UploadFile, session_id: str = Form(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    sessions.create(session_id, img)
    return {"session_id": session_id, "image_size": img.size}


@app.post("/detect/session/{session_id}")
async def detect_in_session(
    session_id: str,
    caption: str = Form(...),
    box_threshold: float = Form(default=0.35),
    text_threshold: float = Form(default=0.25),
):
    sess = sessions.get(session_id)
    img = sess["image"]
    
    result = detector.detect(img, text_prompt=caption)
    
    # 在图像上绘制检测框
    overlay = detector.draw_boxes(img, result["boxes"])
    img_bytes = io.BytesIO()
    overlay.save(img_bytes, format="PNG")
    
    return result


@app.delete("/detect/session/{session_id}")
async def delete_session(session_id: str):
    sessions.delete(session_id)
    return {"message": f"Session {session_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8266))
    uvicorn.run(app, host="0.0.0.0", port=port)
