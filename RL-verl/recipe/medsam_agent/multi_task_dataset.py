"""
多任务 Dataset
==============
继承 RLHFDataset, 从 parquet 加载多任务数据。

parquet 列:
  - images: [{"bytes": ...}]
  - extra_info: {task_type, target_description, label, sample_id}
  - reward_model: {ground_truth, gt_bbox, gt_label, task_type}
"""

import io
import json
import logging
import numpy as np
from PIL import Image
import torch
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


class MultiTaskDataset(RLHFDataset):

    def __init__(self, *args, qwen_image_size=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.qwen_image_size = qwen_image_size

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        image_key = "images"

        # 获取任务信息和目标描述
        extra = row_dict.get("extra_info", {})
        task_type = extra.get("task_type", "segment")
        target = extra.get("target_description", "breast tumor")
        label = extra.get("label", "unknown")

        # 加载图像
        images = []
        original_images = []
        images_data = row_dict.get(image_key)
        if images_data is not None:
            if isinstance(images_data, np.ndarray):
                images_data = images_data.tolist()
            if isinstance(images_data, list):
                for img_item in images_data:
                    if isinstance(img_item, dict) and "bytes" in img_item:
                        orig = Image.open(io.BytesIO(img_item["bytes"])).convert("RGB")
                        original_images.append(orig)
                        img = orig.resize(
                            (self.qwen_image_size, self.qwen_image_size),
                            Image.BILINEAR
                        )
                        images.append(img)

        if not images:
            logger.warning(f"No images for item {item}")
            images = [Image.new("RGB", (self.qwen_image_size, self.qwen_image_size))]
            original_images = images[:]

        # 加载 GT
        reward_data = row_dict.get("reward_model", {})
        gt_mask = None
        gt_bbox = None
        gt_label = None
        if isinstance(reward_data, dict):
            gt_bbox = reward_data.get("gt_bbox")
            gt_label = reward_data.get("gt_label", label)
            gt_raw = reward_data.get("ground_truth")
            if gt_raw and isinstance(gt_raw, dict) and "bytes" in gt_raw:
                gt_mask = Image.open(io.BytesIO(gt_raw["bytes"])).convert("L")

        # 构建 system prompt
        system_prompt = self._build_system_prompt(task_type)

        # 构建 user prompt
        user_text = self._build_user_prompt(task_type, target)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": images[0]},
                {"type": "text", "text": user_text}
            ]}
        ]

        row_dict[self.prompt_key] = messages

        # tools_kwargs (所有工具共享原图)
        if original_images:
            row_dict["tools_kwargs"] = {
                "detect":    {"create_kwargs": {"image": original_images[0]}},
                "classify":  {"create_kwargs": {"image": original_images[0]}},
                "add_bbox":  {"create_kwargs": {"image": original_images[0]}},
                "add_point": {"create_kwargs": {"image": original_images[0]}},
                "stop_action": {"create_kwargs": {"image": original_images[0]}},
            }

        row_dict["agent_name"] = "multi_task_agent"

        # extra_info
        out_extra = dict(extra)
        out_extra.update({
            "task_type": task_type,
            "ground_truth_mask": gt_mask,
            "gt_bbox": gt_bbox,
            "gt_label": gt_label,
            "label": label,
            "reward_model": reward_data,
        })
        row_dict["extra_info"] = out_extra

        # multi_modal_data
        multi_modal_data = {}
        if images:
            multi_modal_data["image"] = images
        row_dict["multi_modal_data"] = multi_modal_data

        # Processor
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            else:
                image_inputs, video_inputs = process_vision_info(messages)

            model_inputs = self.processor(
                text=[raw_prompt], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            model_inputs.pop("second_per_grid_ts", None)

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids, attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True, truncation=self.truncation,
        )

        # position_ids
        if self.processor is not None:
            is_qwen3 = "Qwen3VLProcessor" in self.processor.__class__.__name__
            if is_qwen3:
                from verl.models.transformers.qwen3_vl import get_rope_index
                vision_position_ids = get_rope_index(
                    self.processor, input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    attention_mask=attention_mask[0],
                )
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index
                vision_position_ids = get_rope_index(
                    self.processor, input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            valid_mask = attention_mask[0].bool()
            text_pos = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_pos, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        idx = extra.get("id", item)
        row_dict["index"] = idx
        return row_dict

    # ========== Prompt 构建 ==========

    def _build_system_prompt(self, task_type: str) -> str:
        """构建 system prompt，包含工具定义。"""
        tools_json = json.dumps([
            {"type": "function", "function": {
                "name": "classify",
                "description": "Classify the image or a specific region. Returns label and confidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Classification question"},
                        "region": {"type": "array", "items": {"type": "integer"},
                                   "minItems": 4, "maxItems": 4,
                                   "description": "Optional region [x1,y1,x2,y2] in 0-999"}
                    },
                    "required": ["question"]
                }
            }},
            {"type": "function", "function": {
                "name": "detect",
                "description": "Detect all instances of the target object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target description"},
                        "threshold": {"type": "number", "description": "Confidence threshold (0-1)", "default": 0.25}
                    },
                    "required": ["target"]
                }
            }},
            {"type": "function", "function": {
                "name": "add_bbox",
                "description": "Add bounding box to initialize segmentation.",
                "parameters": {
                    "type": "object",
                    "properties": {"bbox_2d": {"type": "array", "items": {"type": "integer"},
                                                "minItems": 4, "maxItems": 4,
                                                "description": "bbox [x1,y1,x2,y2] in 0-999"}},
                    "required": ["bbox_2d"]
                }
            }},
            {"type": "function", "function": {
                "name": "add_point",
                "description": "Add point to refine mask (positive/negative).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "point_2d": {"type": "array", "items": {"type": "integer"},
                                      "minItems": 2, "maxItems": 2,
                                      "description": "point [x,y] in 0-999"},
                        "point_type": {"type": "string", "enum": ["positive", "negative"]}
                    },
                    "required": ["point_2d", "point_type"]
                }
            }},
            {"type": "function", "function": {
                "name": "stop_action",
                "description": "Finish task and output final result.",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }}
        ])

        return (
            "You are a professional medical image analysis agent. "
            "Use the appropriate tools to complete the requested task.\n\n"
            "# Tools\n\n"
            "<tools>\n" + tools_json + "\n</tools>\n\n"
            "For each tool call, return JSON within <tool_call></tool_call> tags.\n"
            "Coordinates must be in range [0, 999].\n\n"
            "# Decision Guidelines\n"
            "- Classification task: classify then stop\n"
            "- Detection task: detect then stop\n"
            "- Segmentation task: add_bbox, add_point to refine, then stop\n"
            "- Composite task: first classify the whole image for triage. If suspicious, "
            "detect the target, initialize and refine segmentation from the detection box, "
            "then classify the segmented ROI using its bounding region before stopping. "
            "A confidently normal triage result may stop without localization.\n\n"
            "Choose the right tool based on the user's request."
        )

    def _build_user_prompt(self, task_type: str, target: str) -> str:
        """根据任务类型构建 user prompt。"""
        prompts = {
            "classify":  f"Classify this breast ultrasound image: is it benign, malignant, or normal?",
            "detect":    f"Detect all {target}s in this ultrasound image.",
            "segment":   f"Segment the {target} in this ultrasound image.",
            "composite": (
                f"Analyze this breast ultrasound for {target}. First perform whole-image triage, "
                "then localize and segment any suspicious lesion, and finally characterize the segmented ROI."
            ),
        }
        return prompts.get(task_type, f"Analyze this breast ultrasound image.")
