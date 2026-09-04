"""
多任务 Agent 循环
===============
继承 MedSAMIterativeAgentLoop, 扩展支持分类/检测/分割多种工具。

关键改动: _handle_processing_tools_state() 根据工具类型返回不同反馈:
  - detect:   返回检测框 overlay + 文本
  - classify: 只返回文本
  - add_bbox/add_point: 返回 mask overlay (同原有逻辑)
  - stop_action: 终止
"""

import asyncio
import json
import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from verl.experimental.agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.tools.schemas import ToolResponse
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register("multi_task_agent")
class MultiTaskAgentLoop(ToolAgentLoop):
    """多任务 Agent 循环: 支持分类/检测/分割/复合任务。
    
    工具: detect, classify, add_bbox, add_point, stop_action
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._persistent_instances = {}
        import uuid
        self._episode_id = str(uuid.uuid4())[:8]
        self._latest_mask = None
        self._original_image = None
        self._mask_history = []
        self._detection_results = []
        self._classification_results = []
        self._tool_trace = []

    # ========== 生命周期 ==========

    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """运行一个 episode。"""
        self._persistent_instances = {}
        self._latest_mask = None
        self._mask_history = []
        self._detection_results = []
        self._classification_results = []
        self._tool_trace = []

        original_image_data = kwargs.get("multi_modal_data", {}).get("image", None)
        if isinstance(original_image_data, list) and len(original_image_data) > 0:
            self._original_image = original_image_data[0]
        elif original_image_data is not None:
            self._original_image = original_image_data
        else:
            raise ValueError("Original image required for MultiTaskAgentLoop")

        try:
            output = await super().run(sampling_params, **kwargs)
            stopped = self._check_if_stopped(output)
            output.extra_fields["stopped"] = stopped

            # 保存结果
            if len(self._mask_history) > 0:
                output.extra_fields["pred_mask"] = list(self._mask_history)
            elif self._latest_mask is not None:
                output.extra_fields["pred_mask"] = self._latest_mask
            else:
                output.extra_fields["pred_mask"] = []

            output.extra_fields["detection_boxes"] = self._detection_results
            output.extra_fields["classification_result"] = (
                self._classification_results[-1] if self._classification_results else {}
            )
            output.extra_fields["tool_trace"] = list(self._tool_trace)
            output.extra_fields["action_count"] = len(self._tool_trace)
            if isinstance(self._original_image, Image.Image):
                output.extra_fields["image_size"] = self._original_image.size
            else:
                height, width = self._original_image.shape[:2]
                output.extra_fields["image_size"] = (width, height)

            return output
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """释放所有资源。"""
        import torch, gc
        for tool_name, instance_id in list(self._persistent_instances.items()):
            try:
                if instance_id and tool_name in self.tools:
                    await self.tools[tool_name].release(instance_id)
            except Exception as e:
                logger.warning(f"Cleanup error {tool_name}: {e}")
        self._persistent_instances.clear()
        self._latest_mask = None
        self._original_image = None
        self._mask_history = []
        self._detection_results = []
        self._classification_results = []
        self._tool_trace = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _check_if_stopped(self, output) -> bool:
        if hasattr(output, "response_ids") and output.response_ids:
            text = self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
            return "stop_action" in text
        return False

    # ========== 工具执行管理 ==========

    async def _execute_single_tool_call(self, tool_call, tools_kwargs):
        """执行单个工具调用，管理持久化实例。"""
        import json as json_mod
        tool_name = tool_call.name
        tool_args = {}
        try:
            tool_args = json_mod.loads(tool_call.arguments)
        except Exception:
            pass

        if tool_name not in self.tools:
            return ToolResponse(text=f"Error: unknown tool '{tool_name}'"), -0.1, {}

        tool = self.tools[tool_name]
        kwargs = tools_kwargs.get(tool_name, {})

        # 持久化实例
        if tool_name in self._persistent_instances:
            instance_id = self._persistent_instances[tool_name]
        else:
            instance_id = f"{self._episode_id}_{tool_name}"
            create_kwargs = kwargs.get("create_kwargs", {})
            instance_id, _ = await tool.create(instance_id=instance_id, create_kwargs=create_kwargs)
            self._persistent_instances[tool_name] = instance_id

        response, reward, res = await tool.execute(instance_id, tool_args)
        return response, reward, res

    @staticmethod
    def _parse_tool_arguments(tool_call):
        try:
            return json.loads(tool_call.arguments)
        except (TypeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _response_mask(response):
        if not response.image:
            return None
        images = response.image if isinstance(response.image, list) else [response.image]
        return images[0] if images else None

    # ========== 状态处理 ==========

    async def _handle_processing_tools_state(self, agent_data) -> Any:
        """处理工具执行，根据工具类型生成不同的反馈消息。
        
        这是多任务 Agent 的核心：
        - classify → 文本反馈
        - detect   → 检测框 overlay + 文本
        - add_bbox/add_point → mask overlay + 文本（复用原有逻辑）
        """
        from verl.utils.profiler import simple_timer
        import asyncio as aio
        from verl.experimental.agent_loop.tool_agent_loop import AgentState

        # 检查 stop_action
        is_stop = any(tc.name == "stop_action" for tc in agent_data.tool_calls)
        if is_stop:
            for tool_call in agent_data.tool_calls:
                if tool_call.name == "stop_action":
                    self._tool_trace.append({
                        "turn": len(self._tool_trace) + 1,
                        "tool": "stop_action",
                        "arguments": self._parse_tool_arguments(tool_call),
                        "success": True,
                        "result": {"stop": True},
                        "mask_before": self._latest_mask,
                        "mask_after": self._latest_mask,
                        "tool_reward": 0.0,
                    })
            return AgentState.TERMINATED

        # 执行工具
        tasks = []
        for tc in agent_data.tool_calls[:self.max_parallel_calls]:
            tasks.append(self._execute_single_tool_call(tc, agent_data.tools_kwargs))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await aio.gather(*tasks)

        # 处理响应
        new_mask = None
        response_texts = []
        result_data = {}
        tool_name = agent_data.tool_calls[0].name if agent_data.tool_calls else ""

        for tool_call, (resp, reward, res) in zip(
            agent_data.tool_calls[:self.max_parallel_calls], responses, strict=False
        ):
            if reward is not None:
                agent_data.tool_rewards.append(reward)
            response_texts.append(resp.text or "")
            response_mask = self._response_mask(resp)
            if response_mask is not None:
                new_mask = response_mask
            if isinstance(res, dict):
                result_data.update(res)

            event_result = dict(res) if isinstance(res, dict) else {}
            event_success = event_result.get("success", not (resp.text or "").startswith("Error:"))
            self._tool_trace.append({
                "turn": len(self._tool_trace) + 1,
                "tool": tool_call.name,
                "arguments": self._parse_tool_arguments(tool_call),
                "success": bool(event_success),
                "result": event_result,
                "mask_before": self._latest_mask if tool_call.name in ("add_bbox", "add_point") else None,
                "mask_after": response_mask if tool_call.name in ("add_bbox", "add_point") else None,
                "tool_reward": float(reward or 0.0),
            })

        response_text = "\n".join(response_texts)

        # 保存结果
        if tool_name == "detect" and "boxes" in result_data:
            self._detection_results.append({
                "tool": tool_name,
                "boxes": result_data["boxes"]
            })
        elif tool_name == "classify":
            self._classification_results.append(result_data)

        # === 根据工具类型构造不同的反馈 ===

        if tool_name == "classify":
            # 分类: 只返回文本
            user_msg = {
                "role": "user",
                "content": [{"type": "text",
                    "text": f"{response_text}\nBased on this result, what is your next action?"}]
            }
        elif tool_name in ("add_bbox", "add_point"):
            # 分割工具: mask overlay
            if new_mask is None:
                return AgentState.TERMINATED
            self._latest_mask = new_mask
            self._mask_history.append(new_mask)
            overlay = self._create_overlay(self._original_image, mask=new_mask)
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": f"{response_text}\nWhat is your next action?"}
                ]
            }
        elif tool_name == "detect":
            # 检测: 检测框 overlay
            overlay = self._create_overlay(
                self._original_image,
                boxes=result_data.get("boxes"),
                boxes_normalized=True,
            )
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image", "image": overlay},
                    {"type": "text", "text": f"{response_text}\nWhat is your next action?"}
                ]
            }
        else:
            # 未知工具: 文本反馈
            user_msg = {
                "role": "user",
                "content": [{"type": "text", "text": f"{response_text}\nWhat is your next action?"}]
            }

        agent_data.messages.append(user_msg)

        # 重新编码整个对话
        if self.processor is not None:
            raw = await self.loop.run_in_executor(
                None, lambda: self.processor.apply_chat_template(
                    agent_data.messages, add_generation_prompt=True,
                    tokenize=False, **self.apply_chat_template_kwargs
                )
            )
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                imgs, vids = process_vision_info(agent_data.messages, image_patch_size=16)
            else:
                imgs, vids = process_vision_info(agent_data.messages)

            agent_data.image_data = imgs if imgs else None
            inputs = self.processor(
                text=[raw], images=imgs, videos=vids,
                do_resize=False, padding=True, return_tensors="pt"
            )
            new_ids = inputs["input_ids"].squeeze(0).tolist()
            old_len = len(agent_data.prompt_ids)
            delta = len(new_ids) - old_len
            agent_data.prompt_ids = new_ids
            agent_data.response_mask += [0] * delta
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * delta

        agent_data.user_turns += 1
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    # ========== 辅助 ==========

    def _create_overlay(self, image, mask=None, boxes=None, boxes_normalized=False):
        """生成 overlay 图。"""
        if image is None:
            return None
        if image.mode != "RGB":
            img = image.convert("RGB").copy()
        else:
            img = image.copy()
        arr = np.array(img)

        if mask is not None:
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            m = np.array(mask.convert("L") if mask.mode != "L" else mask)
            bin_mask = (m > 127).astype(np.uint8)
            green = np.array([0, 255, 0], dtype=np.uint8)
            alpha = 0.5
            arr = (arr * (1 - alpha * bin_mask[:, :, None])
                   + green * alpha * bin_mask[:, :, None]).astype(np.uint8)

        if boxes:
            import cv2
            w, h = image.size
            for b in boxes:
                bx = b.get("bbox", b if isinstance(b, list) else [])
                if len(bx) == 4:
                    if boxes_normalized:
                        x1 = int(round(bx[0] * (w - 1) / 999))
                        y1 = int(round(bx[1] * (h - 1) / 999))
                        x2 = int(round(bx[2] * (w - 1) / 999))
                        y2 = int(round(bx[3] * (h - 1) / 999))
                    else:
                        x1, y1, x2, y2 = [int(v) for v in bx]
                    cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return Image.fromarray(arr)
