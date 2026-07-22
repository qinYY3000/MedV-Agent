"""
Clinical Process Reward (CPR) — Phase 1 规则版
===============================================
首个多任务视觉 Agent 的过程级奖励框架。

三个创新模块:
  A. Step-Level Clinical Reasoning Score (步骤级临床推理分)
     - 定位准确性 / 信息增益 / 临床一致性 / 决策置信度
  B. Cross-Task Synergy Score (跨任务协同分)
     - 检测→分割 / 分割→分类 / 检测→分类 的信息流评估
  C. Clinical Safety Constraint (临床安全约束)
     - 漏诊/误诊不对称惩罚

公式:
  CPR = λ₁ × Σ_t γ^t × step_score_t
      + λ₂ × synergy_score
      + λ₃ × final_result_score
      - λ₄ × safety_penalty

设计原则:
  1. 不需要额外模型 (纯规则计算)
  2. 不需要额外标注 (用 GT mask/bbox/label)
  3. 步骤级评分可并行计算
  4. 安全惩罚不对称 (漏诊 > 误诊)
"""

import logging
import re
import json
import numpy as np
from PIL import Image
from typing import Any, Optional, List, Dict

logger = logging.getLogger(__name__)


# ============================================================
# 0. 基础工具函数
# ============================================================

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (pred > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    s = pred_bin.sum() + gt_bin.sum()
    return float(2 * inter / s) if s > 0 else (1.0 if inter == 0 else 0.0)


def compute_bbox_iou(box_a: list, box_b: list) -> float:
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_coverage(bbox: list, mask: np.ndarray) -> float:
    """bbox 覆盖 GT mask 的比例 (召回率)。"""
    if mask is None or len(bbox) != 4:
        return 0.0
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2 = min(x2, w); y2 = min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    mask_region = mask[y1:y2, x1:x2]
    gt_in_box = (mask_region > 127).sum()
    gt_total = (mask > 127).sum()
    return float(gt_in_box / gt_total) if gt_total > 0 else 0.0


def point_in_error_region(point: list, pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    """判断点击是否落在错误区域，返回 (在FN区域, 在FP区域)。"""
    if pred_mask is None or gt_mask is None or len(point) != 2:
        return False, False
    h, w = pred_mask.shape[:2]
    px = min(w - 1, max(0, int(point[0] * w / 999)))
    py = min(h - 1, max(0, int(point[1] * h / 999)))
    pred_val = pred_mask[py, px] > 127
    gt_val = gt_mask[py, px] > 127
    in_fn = (not pred_val) and gt_val    # 假阴性区域
    in_fp = pred_val and (not gt_val)    # 假阳性区域
    return in_fn, in_fp


def extract_tool_calls(solution_str: str) -> List[dict]:
    """从 solution_str 提取所有 tool_call。"""
    pattern = r"<tool_call>(.*?)</tool_call>"
    calls = re.findall(pattern, solution_str, re.DOTALL)
    results = []
    for c in calls:
        try:
            results.append(json.loads(c.strip()))
        except:
            pass
    return results


# ============================================================
# 模块 A: Step-Level Clinical Reasoning Score
# ============================================================

def compute_step_scores(
    tool_calls: List[dict],
    iou_per_turn: List[float],
    pred_masks: List[Any],
    gt_mask: np.ndarray,
    gt_bbox: Optional[list],
    task_type: str,
    detection_boxes: List[dict],
    classification_result: dict,
    gt_label: str,
) -> List[dict]:
    """对每一轮操作计算四维临床推理分。
    
    返回: [{"turn": 1, "tool": "add_bbox", "localization": 0.8, 
            "info_gain": 0.3, "consistency": 1.0, "confidence": 0.7, "step_score": 0.65}, ...]
    """
    step_scores = []
    prev_iou = 0.0
    
    # 临床流程顺序: detect → segment(add_bbox/add_point) → classify → stop
    clinical_order = {"detect": 0, "add_bbox": 1, "add_point": 2, "classify": 3, "stop_action": 4}
    max_seen_order = -1
    
    seg_turn = 0  # 分割轮次计数
    
    for t, tc in enumerate(tool_calls):
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        
        # === 维度1: 定位准确性 ===
        localization = 0.5  # 默认中等
        
        if name == "add_bbox":
            bbox_999 = args.get("bbox_2d", [])
            if gt_bbox and len(bbox_999) == 4:
                # 归一化坐标 → 和 GT bbox 比较
                localization = compute_bbox_iou(bbox_999, gt_bbox)
            elif gt_mask is not None:
                # 没有GT bbox时, 用mask覆盖率
                localization = bbox_coverage(bbox_999, gt_mask)
        
        elif name == "add_point":
            point = args.get("point_2d", [])
            ptype = args.get("point_type", "")
            if t < len(pred_masks) and pred_masks[t] is not None:
                # 获取当前预测mask (这一轮操作前的mask)
                cur_mask = np.array(pred_masks[t]) if not isinstance(pred_masks[t], np.ndarray) else pred_masks[t]
                if len(cur_mask.shape) == 3:
                    cur_mask = cur_mask[:, :, 0]
                in_fn, in_fp = point_in_error_region(point, cur_mask, gt_mask)
                if ptype == "positive" and in_fn:
                    localization = 0.9  # 正点落在假阴性区域 → 好
                elif ptype == "negative" and in_fp:
                    localization = 0.9  # 负点落在假阳性区域 → 好
                elif ptype == "positive" and in_fp:
                    localization = 0.1  # 正点落在假阳性区域 → 差
                elif ptype == "negative" and in_fn:
                    localization = 0.1  # 负点落在假阴性区域 → 差
                else:
                    localization = 0.5  # 点在正确区域, 影响不大
        
        elif name == "detect":
            # 检测定位: 检测框是否覆盖GT
            if detection_boxes and gt_bbox:
                best_iou = max(
                    compute_bbox_iou(b.get("bbox", []), gt_bbox) 
                    for b in detection_boxes if len(b.get("bbox", [])) == 4
                )
                localization = best_iou
            elif gt_mask is not None and detection_boxes:
                best_cov = max(
                    bbox_coverage(b.get("bbox", []), gt_mask)
                    for b in detection_boxes if len(b.get("bbox", [])) == 4
                )
                localization = best_cov
        
        elif name == "classify":
            # 分类定位: 如果指定了region, 检查region是否覆盖GT
            region = args.get("region")
            if region and gt_mask is not None:
                localization = bbox_coverage(region, gt_mask)
            else:
                localization = 0.7  # 整图分类, 默认较高
        
        # === 维度2: 信息增益 ===
        info_gain = 0.0
        
        if name in ("add_bbox", "add_point") and t < len(iou_per_turn):
            cur_iou = iou_per_turn[t] if t < len(iou_per_turn) else prev_iou
            delta = cur_iou - prev_iou
            if delta > 0:
                info_gain = min(1.0, delta * 5)  # 放大, 因为IoU增量通常很小
            elif delta < 0:
                info_gain = max(-0.5, delta * 3)  # 负向信息增益 (让情况变差)
            prev_iou = cur_iou
            seg_turn += 1
        
        elif name == "detect":
            # 检测的信息增益: 是否发现了目标
            if detection_boxes and len(detection_boxes) > 0:
                info_gain = 0.8  # 检测到目标 = 高信息增益
            else:
                info_gain = -0.2  # 没检测到 = 负
        
        elif name == "classify":
            # 分类的信息增益: 是否改变了诊断不确定性
            conf = classification_result.get("confidence", 0.5)
            all_probs = classification_result.get("all_probs", {})
            if all_probs:
                # 用熵衡量不确定性
                probs = np.array(list(all_probs.values()))
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                max_entropy = np.log(len(probs))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                info_gain = 1.0 - normalized_entropy  # 熵越低信息增益越大
            else:
                info_gain = conf
        
        elif name == "stop_action":
            # 停止操作的信息增益: 如果当前IoU已经很高, 停止是正确的
            if iou_per_turn and iou_per_turn[-1] > 0.8:
                info_gain = 0.5  # 在高IoU时停止 = 正面
            elif iou_per_turn and iou_per_turn[-1] < 0.3:
                info_gain = -0.3  # 在低IoU时停止 = 负面
        
        # === 维度3: 临床一致性 (流程顺序) ===
        consistency = 1.0
        cur_order = clinical_order.get(name, 99)
        if cur_order < max_seen_order:
            # 出现了逆序操作 (比如分类后又回去分割)
            consistency = 0.3
        max_seen_order = max(max_seen_order, cur_order)
        
        # 特殊规则: 复合任务中跳过分割直接分类
        if task_type == "composite" and name == "classify":
            has_seg = any(tc2.get("name") in ("add_bbox", "add_point") for tc2 in tool_calls[:t])
            if not has_seg:
                consistency = 0.2  # 复合任务中跳过分割直接分类
        
        # === 维度4: 决策置信度 (简化版: 用信息增益的绝对值) ===
        confidence = min(1.0, abs(info_gain) + 0.3)
        
        # === 步骤综合分 ===
        step_score = (
            0.30 * localization +
            0.35 * max(0, info_gain) +   # 正向信息增益
            0.20 * consistency +
            0.15 * confidence
        )
        
        # 负向信息增益额外扣分
        if info_gain < 0:
            step_score += info_gain * 0.3  # 直接加负值
        
        step_score = max(0.0, min(1.0, step_score))
        
        step_scores.append({
            "turn": t + 1,
            "tool": name,
            "localization": round(localization, 3),
            "info_gain": round(info_gain, 3),
            "consistency": round(consistency, 3),
            "confidence": round(confidence, 3),
            "step_score": round(step_score, 3),
        })
    
    return step_scores


# ============================================================
# 模块 B: Cross-Task Synergy Score
# ============================================================

def compute_synergy_score(
    tool_calls: List[dict],
    detection_boxes: List[dict],
    pred_masks: List[Any],
    classification_result: dict,
    gt_mask: np.ndarray,
    gt_bbox: Optional[list],
) -> dict:
    """评估跨任务工具间的信息流传递。
    
    返回: {"det_seg": float, "seg_cls": float, "det_cls": float, "total": float}
    """
    names = [tc.get("name", "") for tc in tool_calls]
    has_detect = "detect" in names
    has_seg = any(n in ("add_bbox", "add_point") for n in names)
    has_classify = "classify" in names
    
    det_seg_score = 0.0
    seg_cls_score = 0.0
    det_cls_score = 0.0
    
    # --- 检测→分割协同 ---
    if has_detect and has_seg:
        # 检查 add_bbox 的坐标是否复用了 detect 的结果
        det_idx = names.index("detect")
        seg_bboxes = []
        for i, tc in enumerate(tool_calls):
            if tc.get("name") == "add_bbox":
                seg_bboxes.append((i, tc.get("arguments", {}).get("bbox_2d", [])))
        
        if seg_bboxes and detection_boxes:
            # 比较 add_bbox 的坐标和 detect 输出的框
            det_boxes_raw = [b.get("bbox", b.get("raw_boxes", [{}])[0] if isinstance(b.get("raw_boxes"), list) else [])
                            for b in detection_boxes]
            det_boxes_clean = [b for b in det_boxes_raw if len(b) == 4]
            
            if det_boxes_clean and seg_bboxes:
                # 第一个 add_bbox 和最佳 detect 框的相似度
                first_seg_bbox = seg_bboxes[0][1]
                if len(first_seg_bbox) == 4 and len(det_boxes_clean) > 0:
                    best_iou = max(compute_bbox_iou(first_seg_bbox, db) for db in det_boxes_clean)
                    det_seg_score = best_iou  # 复用程度 = IoU
                    
                    if best_iou > 0.5:
                        det_seg_score = min(1.0, det_seg_score + 0.2)  # 高复用给bonus
    
    # --- 分割→分类协同 ---
    if has_seg and has_classify:
        # 检查 classify 的 region 是否和分割 mask 对齐
        cls_idx = names.index("classify")
        cls_args = tool_calls[cls_idx].get("arguments", {})
        cls_region = cls_args.get("region")
        
        if cls_region and pred_masks:
            # 最后一个分割 mask
            last_mask = pred_masks[-1] if pred_masks else None
            if last_mask is not None:
                last_mask_np = np.array(last_mask) if not isinstance(last_mask, np.ndarray) else last_mask
                if len(last_mask_np.shape) == 3:
                    last_mask_np = last_mask_np[:, :, 0]
                # 分类的 region 覆盖 mask 的比例
                coverage = bbox_coverage(cls_region, last_mask_np)
                seg_cls_score = coverage
        elif not cls_region and pred_masks:
            # 没指定region但做了整图分类, 检查分类结果是否和分割一致
            # (简化: 如果分割mask存在且分类有结果, 给中等分)
            seg_cls_score = 0.5
        else:
            seg_cls_score = 0.3  # 有分割有分类但没对齐
    
    # --- 检测→分类协同 (跳过分割) ---
    if has_detect and has_classify and not has_seg:
        # 在复合任务中跳过分割直接分类 = 信息断裂
        det_cls_score = -0.3  # 惩罚
    
    # --- 总协同分 ---
    n_pairs = sum([has_detect and has_seg, has_seg and has_classify, has_detect and has_classify])
    if n_pairs == 0:
        total = 0.5  # 单任务, 无协同可评估
    else:
        pair_scores = []
        if has_detect and has_seg:
            pair_scores.append(det_seg_score)
        if has_seg and has_classify:
            pair_scores.append(seg_cls_score)
        if has_detect and has_classify:
            pair_scores.append(det_cls_score)
        total = np.mean(pair_scores) if pair_scores else 0.5
    
    return {
        "det_seg": round(det_seg_score, 3),
        "seg_cls": round(seg_cls_score, 3),
        "det_cls": round(det_cls_score, 3),
        "total": round(float(total), 3),
    }


# ============================================================
# 模块 C: Clinical Safety Constraint
# ============================================================

def compute_safety_penalty(
    task_type: str,
    tool_calls: List[dict],
    detection_boxes: List[dict],
    classification_result: dict,
    pred_masks: List[Any],
    gt_mask: np.ndarray,
    gt_bbox: Optional[list],
    gt_label: str,
    final_iou: float,
) -> dict:
    """临床安全惩罚: 漏诊/误诊不对称惩罚。
    
    医学场景中:
      - 漏诊 (有肿瘤但没检测到) → 重罚
      - 误诊 (没肿瘤但报了肿瘤) → 中罚
      - 误分类 (良性报恶性) → 中罚 (过度治疗)
      - 漏分类 (恶性报良性) → 重罚 (延误治疗)
    
    返回: {"miss_penalty": float, "false_alarm_penalty": float, 
           "misclassification_penalty": float, "total": float}
    """
    miss_penalty = 0.0       # 漏诊
    false_alarm_penalty = 0.0  # 误诊
    misclass_penalty = 0.0   # 误分类
    
    # --- 漏诊/误诊检查 ---
    if gt_label != "normal":
        # GT 有肿瘤
        if not detection_boxes and final_iou < 0.1:
            # Agent 既没检测到也没分割出来
            miss_penalty = 0.5  # 漏诊重罚
        elif final_iou < 0.3:
            # 分割质量很差, 接近漏诊
            miss_penalty = 0.3
        elif final_iou < 0.5:
            miss_penalty = 0.1
    else:
        # GT 是 normal
        if detection_boxes and len(detection_boxes) > 0:
            # Agent 在正常图像上报了肿瘤
            false_alarm_penalty = 0.2
        if classification_result.get("label", "").lower() in ("benign", "malignant"):
            false_alarm_penalty = max(false_alarm_penalty, 0.15)
    
    # --- 误分类检查 (不对称) ---
    if gt_label != "normal" and classification_result:
        pred_label = classification_result.get("label", "").lower()
        gt_label_lower = gt_label.lower()
        
        if pred_label != gt_label_lower:
            if gt_label_lower == "malignant" and pred_label == "benign":
                # 恶性报良性 → 延误治疗 → 重罚
                misclass_penalty = 0.4
            elif gt_label_lower == "benign" and pred_label == "malignant":
                # 良性报恶性 → 过度治疗 → 中罚
                misclass_penalty = 0.2
            elif pred_label == "normal":
                # 有肿瘤报正常 → 漏诊级别
                misclass_penalty = 0.5
    
    total = miss_penalty + false_alarm_penalty + misclass_penalty
    total = min(1.0, total)  # 封顶
    
    return {
        "miss_penalty": round(miss_penalty, 3),
        "false_alarm_penalty": round(false_alarm_penalty, 3),
        "misclassification_penalty": round(misclass_penalty, 3),
        "total": round(total, 3),
    }


# ============================================================
# 传统结果分 (保底)
# ============================================================

def compute_final_result_score(
    task_type: str,
    iou_per_turn: List[float],
    dice_per_turn: List[float],
    detection_boxes: List[dict],
    classification_result: dict,
    gt_bbox: Optional[list],
    gt_label: str,
) -> float:
    """传统结果导向的奖励 (作为保底)。"""
    if task_type == "classify":
        pred_label = classification_result.get("label", "").lower()
        return 1.0 if pred_label == gt_label.lower() else 0.0
    
    elif task_type == "detect":
        if not detection_boxes or gt_bbox is None:
            return 0.0
        bboxes = [b.get("bbox", []) for b in detection_boxes if len(b.get("bbox", [])) == 4]
        if not bboxes:
            return 0.0
        max_iou = max(compute_bbox_iou(b, gt_bbox) for b in bboxes)
        false_alarm = max(0, len(bboxes) - 1) * 0.1
        return max(0.0, min(1.0, max_iou - false_alarm))
    
    elif task_type == "segment":
        if not iou_per_turn:
            return 0.0
        final_iou = iou_per_turn[-1]
        final_dice = dice_per_turn[-1] if dice_per_turn else final_iou
        return 0.5 * final_iou + 0.5 * final_dice
    
    elif task_type == "composite":
        # 复合: 各子任务加权
        det_r = 0.0
        if detection_boxes and gt_bbox:
            bboxes = [b.get("bbox", []) for b in detection_boxes if len(b.get("bbox", [])) == 4]
            if bboxes:
                det_r = max(compute_bbox_iou(b, gt_bbox) for b in bboxes)
        
        seg_r = (0.5 * iou_per_turn[-1] + 0.5 * dice_per_turn[-1]) if iou_per_turn else 0.0
        
        cls_r = 1.0 if classification_result.get("label", "").lower() == gt_label.lower() else 0.0
        
        return 0.2 * det_r + 0.5 * seg_r + 0.3 * cls_r
    
    return 0.0


# ============================================================
# 格式分
# ============================================================

def compute_format_reward(solution_str: str, task_type: str) -> float:
    tool_calls = extract_tool_calls(solution_str)
    if not tool_calls:
        return 0.0
    
    interaction_tools = {"detect", "classify", "add_bbox", "add_point"}
    has_interaction = any(tc.get("name", "") in interaction_tools for tc in tool_calls)
    ends_with_stop = tool_calls[-1].get("name", "") == "stop_action"
    
    if has_interaction and ends_with_stop:
        return 1.0
    elif has_interaction or ends_with_stop:
        return 0.5
    return 0.0


# ============================================================
# 主函数: CPR 奖励计算
# ============================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs
) -> dict:
    """Clinical Process Reward (CPR) 主入口。
    
    CPR = λ₁ × Σ_t γ^t × step_score_t      (模块A: 步骤级临床推理)
        + λ₂ × synergy_score                (模块B: 跨任务协同)
        + λ₃ × final_result_score           (传统结果分)
        - λ₄ × safety_penalty               (模块C: 安全惩罚)
    
    格式分作为稀疏约束单独加权。
    """
    extra_info = extra_info or {}
    
    # ===== 超参数 =====
    # CPR 权重
    lambda_step = 0.35      # 步骤级推理分权重
    lambda_synergy = 0.15   # 跨任务协同分权重
    lambda_result = 0.35    # 传统结果分权重
    lambda_safety = 0.15    # 安全惩罚权重
    format_weight = 0.15    # 格式分权重 (总和不要求=1, 因为安全是减项)
    
    # 折扣因子 (越早的步骤权重越高)
    gamma = 0.9
    
    # ===== 提取信息 =====
    task_type = extra_info.get("task_type", "segment")
    gt_label = extra_info.get("gt_label", "")
    gt_bbox = extra_info.get("gt_bbox")
    
    # GT mask
    gt_mask = None
    gt_raw = extra_info.get("ground_truth_mask") or (
        ground_truth.get("ground_truth") if isinstance(ground_truth, dict) else ground_truth
    )
    if isinstance(gt_raw, dict) and "bytes" in gt_raw:
        import io
        gt_mask = np.array(Image.open(io.BytesIO(gt_raw["bytes"])))
    elif isinstance(gt_raw, Image.Image):
        gt_mask = np.array(gt_raw)
    elif isinstance(gt_raw, np.ndarray):
        gt_mask = gt_raw
    
    if gt_mask is not None:
        if len(gt_mask.shape) == 3:
            gt_mask = gt_mask[:, :, 0]
        if gt_mask.dtype != np.uint8:
            gt_mask = gt_mask.astype(np.uint8)
    
    # 预测 masks
    pred_masks = extra_info.get("pred_mask", [])
    if not isinstance(pred_masks, list):
        pred_masks = [pred_masks] if pred_masks else []
    
    # 检测/分类结果
    detection_boxes = extra_info.get("detection_boxes", [])
    if detection_boxes and isinstance(detection_boxes[0], dict) and "boxes" in detection_boxes[0]:
        # 嵌套结构, 取最后一个
        detection_boxes = detection_boxes[-1].get("boxes", [])
    
    classification_result = extra_info.get("classification_result", {})
    if isinstance(classification_result, list) and classification_result:
        classification_result = classification_result[-1]
    
    # ===== 1. 格式分 =====
    format_reward = compute_format_reward(solution_str, task_type)
    
    # ===== 2. 提取 tool_calls =====
    tool_calls = extract_tool_calls(solution_str)
    
    # ===== 3. 计算 IoU/Dice per turn =====
    iou_per_turn = []
    dice_per_turn = []
    
    if gt_mask is not None:
        for pm in pred_masks:
            if pm is None:
                continue
            pm_np = np.array(pm) if not isinstance(pm, np.ndarray) else pm
            if len(pm_np.shape) == 3:
                pm_np = pm_np[:, :, 0]
            if pm_np.dtype != np.uint8:
                pm_np = pm_np.astype(np.uint8)
            if pm_np.shape != gt_mask.shape:
                pm_pil = Image.fromarray(pm_np).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                )
                pm_np = np.array(pm_pil)
            iou_per_turn.append(compute_iou(pm_np, gt_mask))
            dice_per_turn.append(compute_dice(pm_np, gt_mask))
    
    final_iou = iou_per_turn[-1] if iou_per_turn else 0.0
    final_dice = dice_per_turn[-1] if dice_per_turn else 0.0
    
    # ===== 模块 A: 步骤级临床推理分 =====
    step_scores = compute_step_scores(
        tool_calls=tool_calls,
        iou_per_turn=iou_per_turn,
        pred_masks=pred_masks,
        gt_mask=gt_mask if gt_mask is not None else np.zeros((1, 1)),
        gt_bbox=gt_bbox,
        task_type=task_type,
        detection_boxes=detection_boxes,
        classification_result=classification_result,
        gt_label=gt_label,
    )
    
    # 折扣加权求和
    discounted_step_score = 0.0
    for s in step_scores:
        t = s["turn"] - 1
        discounted_step_score += (gamma ** t) * s["step_score"]
    # 归一化 (几何级数和)
    n_steps = len(step_scores)
    if n_steps > 0:
        geo_sum = (1 - gamma ** n_steps) / (1 - gamma)
        discounted_step_score /= geo_sum
    
    # ===== 模块 B: 跨任务协同分 =====
    synergy = compute_synergy_score(
        tool_calls=tool_calls,
        detection_boxes=detection_boxes,
        pred_masks=pred_masks,
        classification_result=classification_result,
        gt_mask=gt_mask if gt_mask is not None else np.zeros((1, 1)),
        gt_bbox=gt_bbox,
    )
    
    # ===== 传统结果分 =====
    result_score = compute_final_result_score(
        task_type=task_type,
        iou_per_turn=iou_per_turn,
        dice_per_turn=dice_per_turn,
        detection_boxes=detection_boxes,
        classification_result=classification_result,
        gt_bbox=gt_bbox,
        gt_label=gt_label,
    )
    
    # ===== 模块 C: 安全惩罚 =====
    safety = compute_safety_penalty(
        task_type=task_type,
        tool_calls=tool_calls,
        detection_boxes=detection_boxes,
        classification_result=classification_result,
        pred_masks=pred_masks,
        gt_mask=gt_mask if gt_mask is not None else np.zeros((1, 1)),
        gt_bbox=gt_bbox,
        gt_label=gt_label,
        final_iou=final_iou,
    )
    
    # ===== CPR 最终分数 =====
    cpr_reward = (
        lambda_step * discounted_step_score
        + lambda_synergy * synergy["total"]
        + lambda_result * result_score
        - lambda_safety * safety["total"]
    )
    
    # 格式分作为稀疏约束
    final_score = format_weight * format_reward + (1 - format_weight) * cpr_reward
    final_score = max(0.0, min(1.0, final_score))
    
    # ===== 日志 =====
    logger.info(
        f"[CPR] task={task_type} | "
        f"format={format_reward:.2f} step={discounted_step_score:.3f} "
        f"synergy={synergy['total']:.3f} result={result_score:.3f} "
        f"safety={safety['total']:.3f} | "
        f"CPR={cpr_reward:.3f} final={final_score:.3f}"
    )
    
    return {
        "score": float(final_score),
        "cpr_reward": float(cpr_reward),
        "format_reward": float(format_reward),
        "step_score": float(discounted_step_score),
        "step_details": step_scores,
        "synergy_score": synergy,
        "result_score": float(result_score),
        "safety_penalty": safety,
        "iou": float(final_iou),
        "dice": float(final_dice),
        "iou_per_turn": [float(x) for x in iou_per_turn],
        "task_type": task_type,
    }
