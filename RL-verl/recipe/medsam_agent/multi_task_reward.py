"""
多任务奖励函数
=============
为分类、检测、分割、复合任务提供统一的奖励计算。

设计原则:
1. 各任务有独立的奖励, 最后加权组合
2. 格式奖励 + 质量奖励 + 效率惩罚框架通用
3. 复合任务奖励 = 各子任务奖励加权和 + 工具选择 bonus
"""

import logging
import re
import numpy as np
from PIL import Image
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 0. 基础工具函数
# ============================================================

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算 IoU。"""
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter / union) if union > 0 else 1.0 if inter == 0 else 0.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算 Dice。"""
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    return float(2 * inter / (pred_bin.sum() + gt_bin.sum())) \
           if (pred_bin.sum() + gt_bin.sum()) > 0 else 1.0 if inter == 0 else 0.0


def compute_bbox_iou(box_a: list, box_b: list) -> float:
    """计算两个边界框的 IoU。
    
    Args:
        box_a, box_b: [x1, y1, x2, y2]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    
    return inter / union if union > 0 else 0.0


def extract_tool_calls(solution_str: str) -> list:
    """从 solution_str 中提取所有 tool_call。"""
    pattern = r"<tool_call>(.*?)</tool_call>"
    calls = re.findall(pattern, solution_str, re.DOTALL)
    return calls


# ============================================================
# 1. 格式奖励（所有任务通用）
# ============================================================

def compute_format_reward(solution_str: str) -> float:
    """计算格式奖励。
    
    要求:
    - 至少调用一次交互工具（detect/classify/add_bbox/add_point）
    - 最后以 stop_action 结束
    
    Returns: 1.0 (完美) / 0.5 (部分) / 0.0 (无)
    """
    tool_calls = extract_tool_calls(solution_str)
    
    if not tool_calls:
        return 0.0
    
    import json
    interaction_tools = {"detect", "classify", "add_bbox", "add_point"}
    
    has_interaction = False
    ends_with_stop = False
    
    for call_str in tool_calls:
        try:
            data = json.loads(call_str.strip())
            name = data.get("name", "")
            if name in interaction_tools:
                has_interaction = True
        except json.JSONDecodeError:
            if any(t in call_str for t in interaction_tools):
                has_interaction = True
    
    # 检查最后一个
    try:
        last = json.loads(tool_calls[-1].strip())
        if last.get("name") == "stop_action":
            ends_with_stop = True
    except json.JSONDecodeError:
        if "stop_action" in tool_calls[-1]:
            ends_with_stop = True
    
    if has_interaction and ends_with_stop:
        return 1.0
    elif has_interaction or ends_with_stop:
        return 0.5
    return 0.0


# ============================================================
# 2. 分类奖励
# ============================================================

def compute_classification_reward(extra_info: dict) -> float:
    """分类奖励: 正确=1.0, 错误=0.0。
    
    extra_info 中应有:
        classification_result: {"label": str, "confidence": float}
        gt_label: str
    """
    gt_label = extra_info.get("gt_label", "")
    cls_result = extra_info.get("classification_result", {})
    pred_label = cls_result.get("label", "")
    
    if not pred_label:
        return 0.0
    
    correct = (pred_label.lower() == gt_label.lower())
    return 1.0 if correct else 0.0


# ============================================================
# 3. 检测奖励
# ============================================================

def compute_detection_reward(extra_info: dict, ground_truth: dict) -> float:
    """检测奖励: IoU-based。
    
    检测框 vs GT 框的 IoU, 加上漏检/误检惩罚。
    
    extra_info 中应有:
        detection_boxes: [{"bbox": [x1,y1,x2,y2], ...}, ...]
    ground_truth 中:
        gt_bbox: [x1, y1, x2, y2]
    """
    pred_boxes = extra_info.get("detection_boxes", [])
    gt_bbox = ground_truth.get("gt_bbox")
    
    if gt_bbox is None:
        return 0.0
    
    if len(pred_boxes) == 0:
        return 0.0  # 完全漏检
    
    # BUSI 是单目标, 取和 GT 最匹配的框
    bboxes = []
    for item in pred_boxes:
        bboxes.append(item.get("bbox", item if isinstance(item, list) else []))
    
    max_iou = max(compute_bbox_iou(b, gt_bbox) for b in bboxes if len(b) == 4)
    
    # 误检惩罚
    false_alarm = max(0, len(bboxes) - 1) * 0.1
    
    return max(0.0, min(1.0, max_iou - false_alarm))


# ============================================================
# 4. 分割奖励（复用 MedSAM-Agent 逻辑）
# ============================================================

def compute_segmentation_reward(extra_info: dict, ground_truth: Any) -> float:
    """分割奖励: IoU + Dice + 过冲/步数惩罚 + 改进奖励。
    
    复用 MedSAM-Agent 的奖励设计。
    """
    pred_masks = extra_info.get("pred_mask", [])
    
    if not isinstance(pred_masks, list) or len(pred_masks) == 0:
        return 0.0
    
    # 提取 GT mask
    gt_mask = ground_truth.get("ground_truth") if isinstance(ground_truth, dict) else ground_truth
    if gt_mask is None:
        return 0.0
    
    if isinstance(gt_mask, dict) and "bytes" in gt_mask:
        gt_np = np.array(Image.open(io.BytesIO(gt_mask["bytes"])))
    elif isinstance(gt_mask, Image.Image):
        gt_np = np.array(gt_mask)
    elif isinstance(gt_mask, np.ndarray):
        gt_np = gt_mask
    else:
        return 0.0
    
    if len(gt_np.shape) == 3:
        gt_np = gt_np[:, :, 0]
    if gt_np.dtype != np.uint8:
        gt_np = gt_np.astype(np.uint8)
    
    # 对每轮 mask 计算 IoU
    iou_per_turn = []
    import io
    for turn_mask in pred_masks:
        if isinstance(turn_mask, Image.Image):
            turn_np = np.array(turn_mask)
        elif isinstance(turn_mask, np.ndarray):
            turn_np = turn_mask
        else:
            continue
        
        if len(turn_np.shape) == 3:
            turn_np = turn_np[:, :, 0]
        if turn_np.dtype != np.uint8:
            turn_np = turn_np.astype(np.uint8)
        
        # resize if needed
        if turn_np.shape != gt_np.shape:
            turn_pil = Image.fromarray(turn_np).resize(
                (gt_np.shape[1], gt_np.shape[0]), Image.NEAREST
            )
            turn_np = np.array(turn_pil)
        
        iou = compute_iou(turn_np, gt_np)
        iou_per_turn.append(iou)
    
    if not iou_per_turn:
        return 0.0
    
    base = iou_per_turn[-1]
    
    # 过冲惩罚
    overshoot = 0.0
    if len(iou_per_turn) > 1:
        max_iou = max(iou_per_turn)
        final_iou = iou_per_turn[-1]
        if max_iou - final_iou > 1e-4:
            overshoot = (max_iou - final_iou) * 1.0
    
    # 步数惩罚
    steps = len(iou_per_turn)
    turn_penalty = steps * 0.01
    
    # 改进奖励
    improvement = 0.0
    for i in range(1, len(iou_per_turn)):
        delta = iou_per_turn[i] - iou_per_turn[i-1]
        if delta > 0:
            improvement += delta * 0.1
    
    quality = max(0.0, min(1.0, base - overshoot - turn_penalty + improvement))
    return quality


# ============================================================
# 5. 复合任务奖励
# ============================================================

def compute_composite_reward(
    extra_info: dict, ground_truth: Any, solution_str: str
) -> float:
    """复合任务奖励 = 检测 + 分割 + 分类 加权 + 工具选择 bonus。
    
    weights:
        检测: 0.2 (只需定位)
        分割: 0.5 (核心: 精度)
        分类: 0.3 (最终判断)
    """
    det_r = compute_detection_reward(extra_info, ground_truth)
    seg_r = compute_segmentation_reward(extra_info, ground_truth)
    cls_r = compute_classification_reward(extra_info)
    
    combined = 0.2 * det_r + 0.5 * seg_r + 0.3 * cls_r
    
    # 工具选择奖励: 正确流程 detect→segment→classify 给 bonus
    tool_calls = extract_tool_calls(solution_str)
    import json
    names = []
    for tc in tool_calls:
        try:
            names.append(json.loads(tc.strip()).get("name", ""))
        except:
            pass
    
    # 优先奖励: detect → segment(add_bbox/add_point) → classify → stop
    has_detect = "detect" in names
    has_seg = any(t in names for t in ["add_bbox", "add_point"])
    has_classify = "classify" in names
    
    if has_detect and has_seg and has_classify:
        order_score = 0.05
    elif has_detect and has_seg:
        order_score = 0.03
    else:
        order_score = 0.0
    
    return max(0.0, min(1.0, combined + order_score))


# ============================================================
# 6. 主入口
# ============================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs
) -> dict:
    """统一奖励计算入口。
    
    Returns:
        {"score": float, "format": float, "quality": float, ...}
    """
    extra_info = extra_info or {}
    task_type = extra_info.get("task_type", "segment")
    
    # 格式奖励
    format_r = compute_format_reward(solution_str)
    
    # 质量奖励 (按任务类型)
    gt_data = extra_info.get("reward_model", ground_truth)
    if isinstance(gt_data, dict):
        pass
    elif isinstance(ground_truth, dict):
        gt_data = ground_truth
    
    if task_type == "classify":
        quality_r = compute_classification_reward(extra_info)
    elif task_type == "detect":
        quality_r = compute_detection_reward(extra_info, gt_data)
    elif task_type == "segment":
        quality_r = compute_segmentation_reward(extra_info, gt_data)
    elif task_type == "composite":
        quality_r = compute_composite_reward(extra_info, gt_data, solution_str)
    else:
        quality_r = 0.0
    
    # 效率惩罚
    num_turns = extra_info.get("num_turns", 1)
    efficiency_penalty = min(0.05, num_turns * 0.005)
    
    # 最终分数
    final = 0.2 * format_r + 0.8 * quality_r - efficiency_penalty
    final = max(0.0, min(1.0, final))
    
    logger.debug(
        f"[REWARD] task={task_type}, format={format_r:.3f}, "
        f"quality={quality_r:.3f}, penalty={efficiency_penalty:.3f}, "
        f"final={final:.3f}"
    )
    
    return {
        "score": float(final),
        "format_reward": float(format_r),
        "quality_reward": float(quality_r),
        "task_type": task_type,
    }
