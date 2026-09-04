"""Task-conditioned Clinical Process Reward v2 for multi-tool medical agents.

Single-task reward:
    0.70 * terminal quality + 0.20 * event gain + 0.05 * policy
    + 0.05 * format - action cost - safety penalty

Composite-task reward additionally assigns 0.10 weight to measurable
cross-tool information reuse and uses 0.60 terminal-quality weight.
"""

import io
import json
import logging
import re
from typing import Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
SEGMENT_TOOLS = {"add_bbox", "add_point"}
INTERACTION_TOOLS = {"detect", "classify", *SEGMENT_TOOLS}
PATHOLOGY_LABELS = {
    "benign",
    "malignant",
    "normal",
    "covid",
    "lung_opacity",
    "viral_pneumonia",
    "pneumonia",
}


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(min(upper, max(lower, value)))


def _unwrap(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return value.item()
    return value


def _as_list(value: Any) -> list:
    value = _unwrap(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value)
    return [value]


def _binary_mask(mask: Any) -> Optional[np.ndarray]:
    mask = _unwrap(mask)
    if mask is None:
        return None
    if isinstance(mask, dict) and "bytes" in mask:
        mask = Image.open(io.BytesIO(mask["bytes"]))
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
    elif not isinstance(mask, np.ndarray):
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.dtype == bool or (mask.size and np.nanmax(mask) <= 1):
        return mask.astype(bool)
    return mask > 127


def _resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(image) > 127


def compute_iou(pred: Any, gt: Any) -> float:
    pred_bin = _binary_mask(pred)
    gt_bin = _binary_mask(gt)
    if pred_bin is None or gt_bin is None:
        return 0.0
    pred_bin = _resize_binary(pred_bin, gt_bin.shape)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return float(intersection / union) if union else 1.0


def compute_dice(pred: Any, gt: Any) -> float:
    pred_bin = _binary_mask(pred)
    gt_bin = _binary_mask(gt)
    if pred_bin is None or gt_bin is None:
        return 0.0
    pred_bin = _resize_binary(pred_bin, gt_bin.shape)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    denominator = pred_bin.sum() + gt_bin.sum()
    return float(2 * intersection / denominator) if denominator else 1.0


def compute_bbox_iou(box_a: list, box_b: list) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def extract_tool_calls(solution_str: str) -> list[dict]:
    tool_calls = []
    for payload in TOOL_CALL_PATTERN.findall(solution_str or ""):
        try:
            parsed = json.loads(payload.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
            arguments = parsed.get("arguments", {})
            parsed["arguments"] = arguments if isinstance(arguments, dict) else {}
            tool_calls.append(parsed)
    return tool_calls


def _resolve_image_size(extra_info: dict, gt_mask: Optional[np.ndarray]) -> Optional[tuple[int, int]]:
    image_size = _unwrap(extra_info.get("image_size"))
    if isinstance(image_size, (list, tuple, np.ndarray)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    if gt_mask is not None:
        return int(gt_mask.shape[1]), int(gt_mask.shape[0])
    return None


def normalize_bbox(box: Any, image_size: Optional[tuple[int, int]] = None, tool_space: bool = False) -> list:
    box = _unwrap(box)
    if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
        return []
    values = [float(value) for value in box]
    if max(abs(value) for value in values) <= 1.0:
        normalized = values
    elif tool_space:
        normalized = [value / 999.0 for value in values]
    elif image_size is not None:
        width, height = image_size
        normalized = [
            values[0] / max(width - 1, 1),
            values[1] / max(height - 1, 1),
            values[2] / max(width - 1, 1),
            values[3] / max(height - 1, 1),
        ]
    else:
        normalized = [value / 999.0 for value in values]
    x1, y1, x2, y2 = [_clip(value) for value in normalized]
    return [x1, y1, x2, y2] if x1 < x2 and y1 < y2 else []


def bbox_coverage(bbox: list, mask: Any) -> float:
    mask_bin = _binary_mask(mask)
    if mask_bin is None or len(bbox) != 4:
        return 0.0
    total = mask_bin.sum()
    if not total:
        return 0.0
    height, width = mask_bin.shape
    x1 = max(0, min(width - 1, int(np.floor(bbox[0] * width))))
    y1 = max(0, min(height - 1, int(np.floor(bbox[1] * height))))
    x2 = max(x1 + 1, min(width, int(np.ceil(bbox[2] * width))))
    y2 = max(y1 + 1, min(height, int(np.ceil(bbox[3] * height))))
    return float(mask_bin[y1:y2, x1:x2].sum() / total)


def point_in_error_region(point: list, pred_mask: Any, gt_mask: Any) -> tuple[bool, bool]:
    pred_bin = _binary_mask(pred_mask)
    gt_bin = _binary_mask(gt_mask)
    if pred_bin is None or gt_bin is None or len(point) != 2:
        return False, False
    pred_bin = _resize_binary(pred_bin, gt_bin.shape)
    height, width = gt_bin.shape
    point_x = min(width - 1, max(0, int(round(float(point[0]) * (width - 1) / 999))))
    point_y = min(height - 1, max(0, int(round(float(point[1]) * (height - 1) / 999))))
    pred_value = pred_bin[point_y, point_x]
    gt_value = gt_bin[point_y, point_x]
    return bool(not pred_value and gt_value), bool(pred_value and not gt_value)


def _flatten_detection_boxes(value: Any) -> list[dict]:
    flattened = []
    for item in _as_list(value):
        item = _unwrap(item)
        if isinstance(item, dict) and "boxes" in item:
            flattened.extend(_flatten_detection_boxes(item["boxes"]))
        elif isinstance(item, dict) and len(item.get("bbox", [])) == 4:
            flattened.append(item)
        elif isinstance(item, (list, tuple, np.ndarray)) and len(item) == 4:
            flattened.append({"bbox": list(item)})
    return flattened


def _classification_result(value: Any) -> dict:
    value = _unwrap(value)
    if isinstance(value, list):
        value = value[-1] if value else {}
    return value if isinstance(value, dict) else {}


def _classification_quality(result: dict, gt_label: str) -> float:
    if not gt_label or not result:
        return 0.0
    probabilities = result.get("all_probs", {})
    if isinstance(probabilities, dict):
        normalized = {str(key).lower(): float(value) for key, value in probabilities.items()}
        if gt_label.lower() in normalized:
            return _clip(normalized[gt_label.lower()])
    return 1.0 if str(result.get("label", "")).lower() == gt_label.lower() else 0.0


def _segmentation_quality(mask: Any, gt_mask: Any) -> tuple[float, float, float]:
    if _binary_mask(gt_mask) is None or _binary_mask(mask) is None:
        return 0.0, 0.0, 0.0
    iou = compute_iou(mask, gt_mask)
    dice = compute_dice(mask, gt_mask)
    return 0.5 * iou + 0.5 * dice, iou, dice


def _detection_quality(
    detection_boxes: list[dict],
    gt_bbox: list,
    gt_mask: Any,
    image_size: Optional[tuple[int, int]],
    gt_label: str,
) -> tuple[float, list[list]]:
    normalized_predictions = []
    for item in detection_boxes:
        normalized = normalize_bbox(item.get("bbox", []), tool_space=True)
        if normalized:
            normalized_predictions.append(normalized)
    gt_normalized = normalize_bbox(gt_bbox, image_size=image_size)
    gt_mask_bin = _binary_mask(gt_mask)
    has_target = bool(gt_normalized) or bool(gt_mask_bin is not None and gt_mask_bin.any())
    if gt_label.lower() == "normal":
        has_target = False
    if not has_target:
        return (1.0 if not normalized_predictions else 0.0), normalized_predictions
    if not normalized_predictions:
        return 0.0, normalized_predictions
    if gt_normalized:
        quality = max(compute_bbox_iou(box, gt_normalized) for box in normalized_predictions)
    else:
        quality = max(bbox_coverage(box, gt_mask_bin) for box in normalized_predictions)
    false_positive_penalty = 0.1 * max(0, len(normalized_predictions) - 1)
    return _clip(quality - false_positive_penalty), normalized_predictions


def build_tool_trace(
    tool_calls: list[dict],
    provided_trace: Any,
    pred_masks: list,
    detection_boxes: list[dict],
    classification_result: dict,
    tool_rewards: Any = None,
) -> list[dict]:
    trace = []
    for event in _as_list(provided_trace):
        event = _unwrap(event)
        if isinstance(event, dict):
            trace.append(dict(event))
    if trace:
        return trace
    rewards = _as_list(tool_rewards)
    mask_index = 0
    previous_mask = None
    trace = []
    for index, tool_call in enumerate(tool_calls):
        name = tool_call.get("name", "")
        mask_after = None
        result = {}
        if name in SEGMENT_TOOLS and mask_index < len(pred_masks):
            mask_after = pred_masks[mask_index]
            mask_index += 1
        elif name == "detect":
            result = {"boxes": detection_boxes, "success": True}
        elif name == "classify":
            result = dict(classification_result)
            result.setdefault("success", bool(classification_result))
        elif name == "stop_action":
            result = {"stop": True, "success": True}
        trace.append({
            "turn": index + 1,
            "tool": name,
            "arguments": tool_call.get("arguments", {}),
            "success": bool(result.get("success", True)),
            "result": result,
            "mask_before": previous_mask if name in SEGMENT_TOOLS else None,
            "mask_after": mask_after,
            "tool_reward": float(rewards[index]) if index < len(rewards) else 0.0,
        })
        if name in SEGMENT_TOOLS and mask_after is not None:
            previous_mask = mask_after
    return trace


def compute_format_reward(tool_calls: list[dict], task_type: str) -> float:
    if not tool_calls:
        return 0.0
    names = [call.get("name", "") for call in tool_calls]
    expected = {
        "classify": {"classify"},
        "detect": {"detect"},
        "segment": SEGMENT_TOOLS,
        "composite": INTERACTION_TOOLS,
    }.get(task_type, INTERACTION_TOOLS)
    has_expected_interaction = any(name in expected for name in names)
    ends_with_stop = names[-1] == "stop_action"
    if has_expected_interaction and ends_with_stop:
        return 1.0
    if has_expected_interaction or ends_with_stop:
        return 0.5
    return 0.0


def compute_policy_score(
    tool_calls: list[dict], task_type: str, allow_negative_exit: bool = False
) -> tuple[float, dict]:
    names = [call.get("name", "") for call in tool_calls]
    allowed = {
        "classify": {"classify", "stop_action"},
        "detect": {"detect", "stop_action"},
        "segment": {*SEGMENT_TOOLS, "stop_action"},
        "composite": {*INTERACTION_TOOLS, "stop_action"},
    }.get(task_type, {*INTERACTION_TOOLS, "stop_action"})
    penalty = 0.25 * sum(name not in allowed for name in names)
    missing = []
    if task_type == "classify" and "classify" not in names:
        missing.append("classify")
    elif task_type == "detect" and "detect" not in names:
        missing.append("detect")
    elif task_type == "segment" and not any(name in SEGMENT_TOOLS for name in names):
        missing.append("segment")
    elif task_type == "composite" and not allow_negative_exit:
        if "classify" not in names:
            missing.append("triage_classify")
        if "detect" not in names:
            missing.append("detect")
        if not any(name in SEGMENT_TOOLS for name in names):
            missing.append("segment")
        last_segment_index = max(
            (index for index, name in enumerate(names) if name in SEGMENT_TOOLS), default=-1
        )
        if not any(name == "classify" and index > last_segment_index for index, name in enumerate(names)):
            missing.append("roi_classify")
    penalty += 0.35 * len(missing)

    order_violations = 0
    if task_type == "composite":
        detect_index = names.index("detect") if "detect" in names else None
        segment_indices = [index for index, name in enumerate(names) if name in SEGMENT_TOOLS]
        classify_indices = [index for index, name in enumerate(names) if name == "classify"]
        triage_index = classify_indices[0] if classify_indices else None
        roi_index = next(
            (index for index in classify_indices if segment_indices and index > segment_indices[-1]),
            None,
        )
        if triage_index is not None and detect_index is not None and triage_index > detect_index:
            order_violations += 1
        if detect_index is not None and segment_indices and detect_index > segment_indices[0]:
            order_violations += 1
        if roi_index is not None and segment_indices and roi_index < segment_indices[-1]:
            order_violations += 1
    if "add_point" in names and "add_bbox" in names and names.index("add_point") < names.index("add_bbox"):
        order_violations += 1
    penalty += 0.2 * order_violations
    score = _clip(1.0 - penalty, -1.0, 1.0)
    return score, {
        "missing": missing,
        "order_violations": order_violations,
        "negative_exit": allow_negative_exit,
    }


def compute_synergy_score(
    task_type: str,
    trace: list[dict],
    normalized_detection_boxes: list[list],
    triage_quality: float,
) -> dict:
    if task_type != "composite":
        return {"triage_det": None, "det_seg": None, "seg_roi": None, "total": 0.0}
    components = []
    triage_det = None
    det_seg = None
    seg_roi = None
    detect_turn = next((event["turn"] for event in trace if event.get("tool") == "detect"), None)
    triage_event = next(
        (
            event
            for event in trace
            if event.get("tool") == "classify"
            and (detect_turn is None or event.get("turn", 0) < detect_turn)
        ),
        None,
    )
    if triage_event is not None and detect_turn is not None:
        triage_det = triage_quality
        components.append(triage_det)
    bbox_event = next(
        (
            event
            for event in trace
            if event.get("tool") == "add_bbox" and (detect_turn is None or event.get("turn", 0) > detect_turn)
        ),
        None,
    )
    if bbox_event and normalized_detection_boxes:
        action_bbox = normalize_bbox(bbox_event.get("arguments", {}).get("bbox_2d", []), tool_space=True)
        if action_bbox:
            det_seg = max(compute_bbox_iou(action_bbox, box) for box in normalized_detection_boxes)
            components.append(det_seg)

    final_segment_event = next(
        (event for event in reversed(trace) if event.get("tool") in SEGMENT_TOOLS),
        None,
    )
    final_mask = final_segment_event.get("mask_after") if final_segment_event else None
    roi_event = next(
        (
            event
            for event in trace
            if event.get("tool") == "classify"
            and final_segment_event is not None
            and event.get("turn", 0) > final_segment_event.get("turn", 0)
        ),
        None,
    )
    if roi_event and final_mask is not None:
        region = normalize_bbox(roi_event.get("arguments", {}).get("region", []), tool_space=True)
        if region:
            seg_roi = bbox_coverage(region, final_mask)
            components.append(seg_roi)
    total = float(np.mean(components)) if components else 0.0
    return {
        "triage_det": None if triage_det is None else round(triage_det, 4),
        "det_seg": None if det_seg is None else round(det_seg, 4),
        "seg_roi": None if seg_roi is None else round(seg_roi, 4),
        "total": round(total, 4),
    }


def compute_process_score(
    trace: list[dict],
    gt_mask: Any,
    detection_quality: float,
    gt_label: str,
    terminal_quality: float,
) -> tuple[float, list[dict]]:
    details = []
    scores = []
    for event in trace:
        name = event.get("tool", "")
        success = bool(event.get("success", True))
        if not success:
            gain = -1.0
        elif name in SEGMENT_TOOLS:
            before_quality = _segmentation_quality(event.get("mask_before"), gt_mask)[0]
            after_quality = _segmentation_quality(event.get("mask_after"), gt_mask)[0]
            gain = after_quality - before_quality
        elif name == "detect":
            gain = 2 * detection_quality - 1
        elif name == "classify":
            classification_quality = _classification_quality(
                _classification_result(event.get("result", {})), gt_label
            )
            gain = 2 * classification_quality - 1
        elif name == "stop_action":
            gain = 0.5 if terminal_quality >= 0.8 else (-0.5 if terminal_quality < 0.4 else 0.0)
        else:
            gain = -0.5
        gain = _clip(gain, -1.0, 1.0)
        details.append({"turn": event.get("turn"), "tool": name, "gain": round(gain, 4)})
        scores.append(gain)
    return (float(np.mean(scores)) if scores else -1.0), details


def compute_action_cost(
    trace: list[dict],
    task_type: str,
    gt_mask: Any,
) -> tuple[float, dict]:
    allowed = {
        "classify": {"classify"},
        "detect": {"detect"},
        "segment": SEGMENT_TOOLS,
        "composite": INTERACTION_TOOLS,
    }.get(task_type, INTERACTION_TOOLS)
    base_cost = 0.0
    repeated_cost = 0.0
    irrelevant_cost = 0.0
    failure_cost = 0.0
    unnecessary_refinement_cost = 0.0
    previous_signature = None
    for event in trace:
        name = event.get("tool", "")
        if name == "stop_action":
            continue
        base_cost += 0.01
        signature = (name, json.dumps(event.get("arguments", {}), sort_keys=True))
        if signature == previous_signature:
            repeated_cost += 0.03
        previous_signature = signature
        if name not in allowed:
            irrelevant_cost += 0.05
        if not event.get("success", True):
            failure_cost += max(0.1, abs(float(event.get("tool_reward", 0.0))))
        if name in SEGMENT_TOOLS:
            before_quality = _segmentation_quality(event.get("mask_before"), gt_mask)[0]
            if before_quality >= 0.85:
                unnecessary_refinement_cost += 0.03
    total = min(0.6, base_cost + repeated_cost + irrelevant_cost + failure_cost + unnecessary_refinement_cost)
    return total, {
        "base": round(base_cost, 4),
        "repeated": round(repeated_cost, 4),
        "irrelevant": round(irrelevant_cost, 4),
        "failure": round(failure_cost, 4),
        "unnecessary_refinement": round(unnecessary_refinement_cost, 4),
        "total": round(total, 4),
    }


def compute_safety_penalty(
    task_type: str,
    gt_label: str,
    classification_result: dict,
    detection_quality: float,
    segmentation_quality: float,
    detection_boxes: list[dict],
    gt_mask: Any,
    gt_bbox: list,
) -> dict:
    classification_penalty = 0.0
    detection_penalty = 0.0
    segmentation_penalty = 0.0
    gt_label_lower = gt_label.lower()
    pred_label = str(classification_result.get("label", "")).lower()

    if task_type in {"classify", "composite"} and gt_label_lower in PATHOLOGY_LABELS and pred_label:
        if gt_label_lower == "malignant" and pred_label == "normal":
            classification_penalty = 0.7
        elif gt_label_lower == "malignant" and pred_label == "benign":
            classification_penalty = 0.5
        elif gt_label_lower == "benign" and pred_label == "malignant":
            classification_penalty = 0.25
        elif gt_label_lower != "normal" and pred_label == "normal":
            classification_penalty = 0.4
        elif pred_label != gt_label_lower:
            classification_penalty = 0.15

    gt_mask_bin = _binary_mask(gt_mask)
    has_target = bool(gt_bbox) or bool(gt_mask_bin is not None and gt_mask_bin.any())
    if gt_label_lower == "normal":
        has_target = False
    if task_type in {"detect", "composite"}:
        if has_target and detection_quality < 0.1:
            detection_penalty = 0.5
        elif has_target and detection_quality < 0.3:
            detection_penalty = 0.3
        elif not has_target and detection_boxes:
            detection_penalty = 0.25
    if task_type in {"segment", "composite"}:
        if has_target and segmentation_quality < 0.1:
            segmentation_penalty = 0.4
        elif has_target and segmentation_quality < 0.3:
            segmentation_penalty = 0.25
        elif has_target and segmentation_quality < 0.5:
            segmentation_penalty = 0.1

    total = min(0.8, classification_penalty + detection_penalty + segmentation_penalty)
    return {
        "classification": round(classification_penalty, 4),
        "detection": round(detection_penalty, 4),
        "segmentation": round(segmentation_penalty, 4),
        "total": round(total, 4),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    extra_info = dict(extra_info or {})
    task_type = str(extra_info.get("task_type", "segment")).lower()
    gt_label = str(extra_info.get("gt_label", extra_info.get("label", "")))
    gt_bbox = _unwrap(extra_info.get("gt_bbox", []))
    if not isinstance(gt_bbox, (list, tuple, np.ndarray)):
        gt_bbox = []

    gt_raw = extra_info.get("ground_truth_mask")
    if gt_raw is None:
        gt_raw = ground_truth.get("ground_truth") if isinstance(ground_truth, dict) else ground_truth
    gt_mask = _binary_mask(gt_raw)
    image_size = _resolve_image_size(extra_info, gt_mask)

    pred_masks = _as_list(extra_info.get("pred_mask", []))
    detection_boxes = _flatten_detection_boxes(extra_info.get("detection_boxes", []))
    classification_result = _classification_result(extra_info.get("classification_result", {}))
    tool_calls = extract_tool_calls(solution_str)
    trace = build_tool_trace(
        tool_calls=tool_calls,
        provided_trace=extra_info.get("tool_trace"),
        pred_masks=pred_masks,
        detection_boxes=detection_boxes,
        classification_result=classification_result,
        tool_rewards=extra_info.get("tool_rewards"),
    )

    final_mask = next(
        (event.get("mask_after") for event in reversed(trace) if event.get("tool") in SEGMENT_TOOLS),
        pred_masks[-1] if pred_masks else None,
    )
    segmentation_quality, final_iou, final_dice = _segmentation_quality(final_mask, gt_mask)
    detection_quality, normalized_detection_boxes = _detection_quality(
        detection_boxes, list(gt_bbox), gt_mask, image_size, gt_label
    )
    classification_events = [event for event in trace if event.get("tool") == "classify"]
    detect_turn = next((event.get("turn", 0) for event in trace if event.get("tool") == "detect"), None)
    triage_event = next(
        (event for event in classification_events if detect_turn is not None and event.get("turn", 0) < detect_turn),
        None,
    )
    triage_result = _classification_result(triage_event.get("result", {})) if triage_event else {}
    triage_quality = _classification_quality(triage_result, gt_label)
    roi_event = next(
        (
            event
            for event in reversed(classification_events)
            if any(
                segment_event.get("tool") in SEGMENT_TOOLS
                and segment_event.get("turn", 0) < event.get("turn", 0)
                for segment_event in trace
            )
        ),
        None,
    )
    roi_result = _classification_result(roi_event.get("result", {})) if roi_event else {}
    classification_quality = _classification_quality(roi_result or classification_result, gt_label)

    if task_type == "classify":
        terminal_quality = classification_quality
        task_components = {"classification": classification_quality}
    elif task_type == "detect":
        terminal_quality = detection_quality
        task_components = {"detection": detection_quality}
    elif task_type == "segment":
        terminal_quality = segmentation_quality
        task_components = {"segmentation": segmentation_quality}
    elif task_type == "composite":
        terminal_quality = (
            0.25 * detection_quality
            + 0.45 * segmentation_quality
            + 0.30 * classification_quality
        )
        task_components = {
            "detection": detection_quality,
            "segmentation": segmentation_quality,
            "classification": classification_quality,
        }
    else:
        terminal_quality = 0.0
        task_components = {}

    process_score, event_details = compute_process_score(
        trace, gt_mask, detection_quality, gt_label, terminal_quality
    )
    synergy = compute_synergy_score(task_type, trace, normalized_detection_boxes, triage_quality)
    allow_negative_exit = task_type == "composite" and gt_label.lower() == "normal"
    policy_score, policy_details = compute_policy_score(tool_calls, task_type, allow_negative_exit)
    format_reward = compute_format_reward(tool_calls, task_type)
    action_cost, action_cost_details = compute_action_cost(trace, task_type, gt_mask)
    safety = compute_safety_penalty(
        task_type=task_type,
        gt_label=gt_label,
        classification_result=classification_result,
        detection_quality=detection_quality,
        segmentation_quality=segmentation_quality,
        detection_boxes=detection_boxes,
        gt_mask=gt_mask,
        gt_bbox=list(gt_bbox),
    )

    if task_type == "composite":
        raw_score = (
            0.60 * terminal_quality
            + 0.20 * process_score
            + 0.10 * synergy["total"]
            + 0.05 * policy_score
            + 0.05 * format_reward
            - action_cost
            - safety["total"]
        )
    else:
        raw_score = (
            0.70 * terminal_quality
            + 0.20 * process_score
            + 0.05 * policy_score
            + 0.05 * format_reward
            - action_cost
            - safety["total"]
        )
    final_score = _clip(raw_score, -1.0, 1.0)

    logger.info(
        "[CPR-v2] task=%s terminal=%.3f process=%.3f synergy=%.3f policy=%.3f "
        "format=%.3f cost=%.3f safety=%.3f score=%.3f",
        task_type,
        terminal_quality,
        process_score,
        synergy["total"],
        policy_score,
        format_reward,
        action_cost,
        safety["total"],
        final_score,
    )

    return {
        "score": final_score,
        "cpr_reward": final_score,
        "version": "cpr_v2",
        "terminal_quality": float(terminal_quality),
        "result_score": float(terminal_quality),
        "task_components": {key: float(value) for key, value in task_components.items()},
        "triage_quality": float(triage_quality),
        "roi_classification_quality": float(classification_quality),
        "process_score": float(process_score),
        "step_score": float(process_score),
        "step_details": event_details,
        "synergy_score": synergy,
        "policy_score": float(policy_score),
        "policy_details": policy_details,
        "format_reward": float(format_reward),
        "action_cost": action_cost_details,
        "safety_penalty": safety,
        "iou": float(final_iou),
        "dice": float(final_dice),
        "task_type": task_type,
    }
