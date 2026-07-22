"""
Grounding DINO 检测工具类
=======================
Verl 框架工具，调用 Grounding DINO API (:8266) 做零样本目标检测。

特点:
- 零样本: 不需要在目标数据集上训练，传入文本描述即可
- Agent 调用示例: detect(target="breast tumor")
"""

import io
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import requests
from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class DetectTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.timeout = config.get("timeout", 60)
        self.api_url = config.get("api_url", "http://localhost:8266")

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=50, pool_maxsize=50,
            max_retries=requests.adapters.Retry(total=3, backoff_factor=0.5),
            pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    async def create(self, instance_id=None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing 'image' in kwargs")
        if not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        files = {"image": ("image.png", img_bytes, "image/png")}
        data = {"session_id": instance_id}
        resp = self.session.post(
            f"{self.api_url}/detect/session/create",
            files=files, data=data, timeout=self.timeout,
            proxies={"http": None, "https": None}
        )
        resp.raise_for_status()
        result = resp.json()

        w, h = result.get("image_size", (0, 0))
        self._instances[instance_id] = {
            "session_id": result["session_id"],
            "image": image,
            "image_size": (w, h),
            "detection_history": []
        }
        logger.debug(f"Detection session {instance_id} created")
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}

        target = parameters.get("target", "breast tumor")
        threshold = float(parameters.get("threshold", 0.25))
        threshold = max(0.0, min(1.0, threshold))

        try:
            data = {"caption": target, "box_threshold": str(threshold)}
            resp = self.session.post(
                f"{self.api_url}/detect/session/{inst['session_id']}",
                data=data, timeout=self.timeout,
                proxies={"http": None, "https": None}
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logger.error(f"DetectTool failed: {e}")
            return ToolResponse(text=f"Error: {str(e)}"), -0.1, {"success": False}

        boxes = result.get("boxes", [])
        n = result.get("num_detections", len(boxes))

        # 生成 overlay
        overlay = self._draw_boxes(inst["image"], boxes)

        # 文本描述
        if n == 0:
            text = f"No '{target}' detected."
        else:
            text = f"Detected {n} region(s):\n"
            for i, b in enumerate(boxes):
                text += f"  [{i+1}] bbox={b['bbox']}, score={b['score']:.3f}"
                if b.get("label"):
                    text += f", label={b['label']}"
                text += "\n"

        inst["detection_history"].append({"target": target, "boxes": boxes})

        # 归一化到 0-999
        w, h = inst["image_size"]
        norm_boxes = []
        if w > 0 and h > 0:
            for b in boxes:
                x1, y1, x2, y2 = b["bbox"]
                norm_boxes.append({
                    "bbox": [int(x1*999/w), int(y1*999/h), int(x2*999/w), int(y2*999/h)],
                    "score": b["score"],
                    "label": b.get("label", "")
                })

        return ToolResponse(image=[overlay], text=text), 0.0, {
            "success": True, "boxes": norm_boxes, "raw_boxes": boxes, "num_detections": n
        }

    async def release(self, instance_id, **kwargs):
        if instance_id in self._instances:
            try:
                self.session.delete(
                    f"{self.api_url}/detect/session/{instance_id}",
                    timeout=5, proxies={"http": None, "https": None}
                )
            except Exception:
                pass
            self._instances.pop(instance_id, None)
        import gc
        gc.collect()

    def _draw_boxes(self, image, boxes):
        from PIL import ImageDraw
        img = image.convert("RGB").copy() if image.mode != "RGB" else image.copy()
        draw = ImageDraw.Draw(img)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            label = f"{b.get('label','obj')} {b['score']:.2f}"
            draw.text((x1+2, y1-15), label, fill="blue")
        return img

    def get_openai_tool_schema(self):
        return self.tool_schema
