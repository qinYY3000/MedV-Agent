import json

import numpy as np

from recipe.medsam_agent.cpr_reward import compute_score


def tool_call(name, arguments=None):
    payload = {"name": name, "arguments": arguments or {}}
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def square_mask(start=20, end=80):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[start:end, start:end] = 255
    return mask


def segmentation_extra(mask, trace):
    return {
        "task_type": "segment",
        "gt_label": "lesion",
        "ground_truth_mask": square_mask(),
        "pred_mask": [mask],
        "tool_trace": trace,
        "image_size": (100, 100),
    }


def test_short_accurate_segmentation_beats_redundant_and_bad_trajectories():
    good_mask = square_mask()
    bad_mask = square_mask(20, 50)
    good_solution = tool_call("add_bbox", {"bbox_2d": [202, 202, 797, 797]}) + tool_call("stop_action")
    good_trace = [
        {"turn": 1, "tool": "add_bbox", "arguments": {"bbox_2d": [202, 202, 797, 797]},
         "success": True, "mask_before": None, "mask_after": good_mask, "tool_reward": 0.0},
        {"turn": 2, "tool": "stop_action", "arguments": {}, "success": True,
         "mask_before": good_mask, "mask_after": good_mask, "tool_reward": 0.0},
    ]
    redundant_solution = (
        tool_call("add_bbox", {"bbox_2d": [202, 202, 797, 797]})
        + tool_call("add_point", {"point_2d": [500, 500], "point_type": "positive"})
        + tool_call("stop_action")
    )
    redundant_trace = [
        good_trace[0],
        {"turn": 2, "tool": "add_point", "arguments": {"point_2d": [500, 500], "point_type": "positive"},
         "success": True, "mask_before": good_mask, "mask_after": good_mask, "tool_reward": 0.0},
        {"turn": 3, "tool": "stop_action", "arguments": {}, "success": True,
         "mask_before": good_mask, "mask_after": good_mask, "tool_reward": 0.0},
    ]
    bad_trace = [
        {"turn": 1, "tool": "add_bbox", "arguments": {"bbox_2d": [202, 202, 500, 500]},
         "success": True, "mask_before": None, "mask_after": bad_mask, "tool_reward": 0.0},
        {"turn": 2, "tool": "stop_action", "arguments": {}, "success": True,
         "mask_before": bad_mask, "mask_after": bad_mask, "tool_reward": 0.0},
    ]

    good = compute_score("test", good_solution, square_mask(), segmentation_extra(good_mask, good_trace))
    redundant = compute_score("test", redundant_solution, square_mask(), segmentation_extra(good_mask, redundant_trace))
    bad = compute_score("test", good_solution, square_mask(), segmentation_extra(bad_mask, bad_trace))

    assert good["score"] > redundant["score"] > bad["score"]
    assert good["iou"] == 1.0
    assert redundant["action_cost"]["unnecessary_refinement"] > 0


def test_detection_uses_consistent_pixel_and_tool_coordinate_spaces():
    solution = tool_call("detect", {"target": "lesion"}) + tool_call("stop_action")
    extra = {
        "task_type": "detect",
        "gt_label": "lesion",
        "gt_bbox": [20, 20, 80, 80],
        "image_size": (100, 100),
        "detection_boxes": [{"bbox": [202, 202, 807, 807], "score": 0.9}],
    }
    result = compute_score("test", solution, {}, extra)

    assert result["terminal_quality"] > 0.95
    assert result["safety_penalty"]["detection"] == 0.0


def test_classification_is_not_penalized_for_missing_detection_or_segmentation():
    solution = tool_call("classify", {"question": "diagnosis"}) + tool_call("stop_action")
    extra = {
        "task_type": "classify",
        "gt_label": "malignant",
        "classification_result": {
            "label": "malignant",
            "confidence": 0.9,
            "all_probs": {"benign": 0.05, "malignant": 0.9, "normal": 0.05},
        },
    }
    result = compute_score("test", solution, {}, extra)

    assert result["terminal_quality"] == 0.9
    assert result["safety_penalty"]["detection"] == 0.0
    assert result["safety_penalty"]["segmentation"] == 0.0
    assert result["score"] > 0.7


def test_malignant_to_benign_is_penalized_more_than_benign_to_malignant():
    solution = tool_call("classify", {"question": "diagnosis"}) + tool_call("stop_action")
    malignant_missed = compute_score(
        "test", solution, {},
        {"task_type": "classify", "gt_label": "malignant", "classification_result": {"label": "benign"}},
    )
    benign_overcalled = compute_score(
        "test", solution, {},
        {"task_type": "classify", "gt_label": "benign", "classification_result": {"label": "malignant"}},
    )

    assert malignant_missed["safety_penalty"]["classification"] > benign_overcalled["safety_penalty"]["classification"]
    assert malignant_missed["score"] < benign_overcalled["score"]


def test_composite_information_reuse_increases_synergy_reward():
    mask = square_mask()
    solution = (
        tool_call("detect", {"target": "tumor"})
        + tool_call("add_bbox", {"bbox_2d": [202, 202, 797, 797]})
        + tool_call("classify", {"question": "diagnosis"})
        + tool_call("stop_action")
    )
    common = {
        "task_type": "composite",
        "gt_label": "malignant",
        "gt_bbox": [20, 20, 80, 80],
        "ground_truth_mask": mask,
        "image_size": (100, 100),
        "detection_boxes": [{"bbox": [202, 202, 797, 797], "score": 0.9}],
        "classification_result": {"label": "malignant", "all_probs": {"malignant": 0.95, "benign": 0.05}},
    }
    aligned_trace = [
        {"turn": 1, "tool": "detect", "arguments": {"target": "tumor"}, "success": True,
         "result": {"boxes": common["detection_boxes"]}, "tool_reward": 0.0},
        {"turn": 2, "tool": "add_bbox", "arguments": {"bbox_2d": [202, 202, 797, 797]}, "success": True,
         "mask_before": None, "mask_after": mask, "tool_reward": 0.0},
        {"turn": 3, "tool": "classify", "arguments": {"question": "diagnosis"}, "success": True,
         "result": common["classification_result"], "tool_reward": 0.0},
        {"turn": 4, "tool": "stop_action", "arguments": {}, "success": True, "tool_reward": 0.0},
    ]
    misaligned_trace = [dict(event) for event in aligned_trace]
    misaligned_trace[1] = dict(misaligned_trace[1])
    misaligned_trace[1]["arguments"] = {"bbox_2d": [0, 0, 150, 150]}

    aligned = compute_score("test", solution, {}, {**common, "tool_trace": aligned_trace})
    misaligned = compute_score("test", solution, {}, {**common, "tool_trace": misaligned_trace})

    assert aligned["synergy_score"]["det_seg"] > misaligned["synergy_score"]["det_seg"]
    assert aligned["score"] > misaligned["score"]


def test_failed_tool_call_receives_direct_negative_signal():
    solution = tool_call("classify", {"question": "diagnosis"}) + tool_call("stop_action")
    successful = {
        "task_type": "classify",
        "gt_label": "benign",
        "classification_result": {"label": "benign"},
        "tool_trace": [
            {"turn": 1, "tool": "classify", "arguments": {"question": "diagnosis"},
             "success": True, "result": {"label": "benign"}, "tool_reward": 0.0},
            {"turn": 2, "tool": "stop_action", "arguments": {}, "success": True, "tool_reward": 0.0},
        ],
    }
    failed = {
        **successful,
        "classification_result": {},
        "tool_trace": [
            {"turn": 1, "tool": "classify", "arguments": {"question": "diagnosis"},
             "success": False, "result": {}, "tool_reward": -0.1},
            {"turn": 2, "tool": "stop_action", "arguments": {}, "success": True, "tool_reward": 0.0},
        ],
    }

    successful_result = compute_score("test", solution, {}, successful)
    failed_result = compute_score("test", solution, {}, failed)

    assert failed_result["action_cost"]["failure"] >= 0.1
    assert failed_result["score"] < successful_result["score"]


def test_normal_detection_with_no_boxes_is_rewarded_when_tool_succeeds():
    solution = tool_call("detect", {"target": "tumor"}) + tool_call("stop_action")
    extra = {
        "task_type": "detect",
        "gt_label": "normal",
        "detection_boxes": [],
        "tool_trace": [
            {"turn": 1, "tool": "detect", "arguments": {"target": "tumor"},
             "success": True, "result": {"boxes": []}, "tool_reward": 0.0},
            {"turn": 2, "tool": "stop_action", "arguments": {}, "success": True, "tool_reward": 0.0},
        ],
    }

    result = compute_score("test", solution, {}, extra)

    assert result["terminal_quality"] == 1.0
    assert result["safety_penalty"]["total"] == 0.0
    assert result["score"] > 0.8


def test_legacy_outputs_build_event_aligned_segmentation_trace():
    first_mask = square_mask(25, 70)
    final_mask = square_mask()
    solution = (
        tool_call("detect", {"target": "lesion"})
        + tool_call("add_bbox", {"bbox_2d": [202, 202, 797, 797]})
        + tool_call("add_point", {"point_2d": [750, 750], "point_type": "positive"})
        + tool_call("stop_action")
    )
    extra = {
        "task_type": "segment",
        "gt_label": "lesion",
        "ground_truth_mask": final_mask,
        "pred_mask": [first_mask, final_mask],
        "detection_boxes": [{"bbox": [202, 202, 797, 797]}],
    }

    result = compute_score("test", solution, {}, extra)
    segment_steps = [step for step in result["step_details"] if step["tool"] in {"add_bbox", "add_point"}]

    assert len(segment_steps) == 2
    assert segment_steps[0]["gain"] > 0
    assert segment_steps[1]["gain"] > 0
    assert result["iou"] == 1.0
