"""轻量级多实例状态机，用于单图多病灶/多目标 Agent 编排。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class MultiInstanceState:
    """维护检测候选、各实例 mask 与实例级完成状态。

    该类不依赖 API 或 Ray，便于单测；Agent loop 负责将 candidate_id 映射到
    独立的分割 session，并将工具结果回写到本状态机。
    """

    def __init__(self) -> None:
        self._instances: dict[str, dict[str, Any]] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._instances.clear()
        self._next_id = 0

    def register_detections(self, boxes: list[dict[str, Any]]) -> list[str]:
        """注册检测框，并为每个候选生成稳定的 candidate_id。"""
        candidate_ids = []
        for box in boxes:
            bbox = box.get("bbox", []) if isinstance(box, dict) else []
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            candidate_id = f"candidate_{self._next_id}"
            self._next_id += 1
            self._instances[candidate_id] = {
                "instance_id": candidate_id,
                "detected_bbox": list(bbox),
                "detection_score": box.get("score") if isinstance(box, dict) else None,
                "detection_label": box.get("label", "") if isinstance(box, dict) else "",
                "mask": None,
                "mask_history": [],
                "classification": None,
                "status": "active",
            }
            candidate_ids.append(candidate_id)
        return candidate_ids

    def resolve(self, instance_id: str | None) -> str | None:
        """解析显式实例 ID；单候选时允许省略 instance_id 以兼容旧轨迹。"""
        if instance_id in self._instances:
            return instance_id
        active_ids = self.pending_ids()
        return active_ids[0] if instance_id is None and len(active_ids) == 1 else None

    def record_mask(self, instance_id: str | None, mask: Any) -> bool:
        resolved_id = self.resolve(instance_id)
        if resolved_id is None or mask is None:
            return False
        instance = self._instances[resolved_id]
        if instance["status"] == "finished":
            return False
        instance["mask"] = mask
        instance["mask_history"].append(mask)
        return True

    def record_classification(self, instance_id: str | None, result: dict[str, Any]) -> bool:
        resolved_id = self.resolve(instance_id)
        if resolved_id is None:
            return False
        self._instances[resolved_id]["classification"] = dict(result)
        return True

    def current_mask(self, instance_id: str | None) -> Any:
        resolved_id = self.resolve(instance_id)
        return self._instances[resolved_id]["mask"] if resolved_id is not None else None

    def finish(self, instance_id: str | None) -> bool:
        resolved_id = self.resolve(instance_id)
        if resolved_id is None:
            return False
        instance = self._instances[resolved_id]
        if instance["status"] == "finished" or instance["mask"] is None:
            return False
        instance["status"] = "finished"
        return True

    def pending_ids(self) -> list[str]:
        return [instance_id for instance_id, instance in self._instances.items() if instance["status"] == "active"]

    def all_finished(self) -> bool:
        return bool(self._instances) and not self.pending_ids()

    def pred_instances(self) -> list[dict[str, Any]]:
        """返回可写入 rollout extra_fields 的实例状态快照。"""
        return [deepcopy(instance) for instance in self._instances.values()]
