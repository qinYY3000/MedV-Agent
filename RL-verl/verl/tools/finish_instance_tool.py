"""实例级完成工具。

该工具不直接调用视觉后端；MultiTaskAgentLoop 接收该动作后，将当前候选实例
从 active 标记为 finished，并继续处理其余候选。"""

from typing import Any

from .base_tool import BaseTool
from .schemas import ToolResponse


class FinishInstanceTool(BaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        target_instance_id = parameters.get("instance_id")
        if not isinstance(target_instance_id, str) or not target_instance_id:
            return ToolResponse(text="Error: missing 'instance_id' parameter"), -0.05, {"success": False}
        return ToolResponse(
            text=f"Requested completion for instance {target_instance_id}."
        ), 0.0, {"success": True, "instance_id": target_instance_id}
