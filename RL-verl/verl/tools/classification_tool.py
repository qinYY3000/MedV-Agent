"""
BioMedCLIP 分类工具类
====================
Verl 框架工具，调用 BioMedCLIP API (:8267) 做零样本医学图像分类。

特点:
- 零样本: 不需要训练，直接比较图像与医学文本嵌入
- 支持整图分类和区域分类 (裁剪后分类)
- 返回文本结果 (不产生 overlay 图)
"""

import io
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import requests
from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class ClassifyTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}
        self.timeout = config.get("timeout", 30)
        self.api_url = config.get("api_url", "http://localhost:8267")

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

        self._instances[instance_id] = {
            "image": image,
            "image_size": image.size,
            "history": []
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        inst = self._instances.get(instance_id)
        if inst is None:
            return ToolResponse(text="Error: invalid instance_id"), -0.1, {"success": False}

        question = parameters.get("question", "Classify this image.")
        region = parameters.get("region")  # [x1,y1,x2,y2] in 0-999
        image = inst["image"].copy()

        try:
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            files = {"image": ("image.png", img_bytes, "image/png")}

            if region is not None and len(region) == 4:
                # 坐标映射: 0-999 → 实际像素
                w, h = inst["image_size"]
                x1 = int(region[0] * w / 999)
                y1 = int(region[1] * h / 999)
                x2 = int(region[2] * w / 999)
                y2 = int(region[3] * h / 999)
                data = {"bbox": json.dumps([x1, y1, x2, y2])}
                resp = self.session.post(
                    f"{self.api_url}/classify/region",
                    files=files, data=data, timeout=self.timeout,
                    proxies={"http": None, "https": None}
                )
            else:
                resp = self.session.post(
                    f"{self.api_url}/classify",
                    files=files, timeout=self.timeout,
                    proxies={"http": None, "https": None}
                )

            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            logger.error(f"ClassifyTool failed: {e}")
            return ToolResponse(text=f"Error: {str(e)}"), -0.1, {"success": False}

        label = result.get("label", "unknown")
        conf = result.get("confidence", 0.0)
        all_probs = result.get("all_probs", {})

        text = f"Classification: {label} (confidence: {conf:.3f})\n"
        if all_probs:
            text += "Probabilities:\n"
            for cls_name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
                text += f"  {cls_name}: {prob:.3f}\n"

        inst["history"].append({"label": label, "confidence": conf})
        return ToolResponse(text=text), 0.0, {
            "success": True, "label": label, "confidence": conf, "all_probs": all_probs
        }

    async def release(self, instance_id, **kwargs):
        self._instances.pop(instance_id, None)
        import gc
        gc.collect()

    def get_openai_tool_schema(self):
        return self.tool_schema
