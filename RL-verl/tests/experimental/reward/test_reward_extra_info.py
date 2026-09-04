import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def load_build_reward_extra_info(monkeypatch):
    verl_module = types.ModuleType("verl")
    verl_module.DataProto = object
    reward_loop_module = types.ModuleType("verl.experimental.reward.reward_loop")
    reward_loop_module.register = lambda _name: lambda cls: cls
    base_module = types.ModuleType("verl.experimental.reward.reward_loop.base")
    base_module.RewardLoopManagerBase = object
    score_module = types.ModuleType("verl.utils.reward_score")
    score_module.default_compute_score = None
    monkeypatch.setitem(sys.modules, "verl", verl_module)
    monkeypatch.setitem(sys.modules, "verl.experimental", types.ModuleType("verl.experimental"))
    monkeypatch.setitem(sys.modules, "verl.experimental.reward", types.ModuleType("verl.experimental.reward"))
    monkeypatch.setitem(sys.modules, "verl.experimental.reward.reward_loop", reward_loop_module)
    monkeypatch.setitem(sys.modules, "verl.experimental.reward.reward_loop.base", base_module)
    monkeypatch.setitem(sys.modules, "verl.utils", types.ModuleType("verl.utils"))
    monkeypatch.setitem(sys.modules, "verl.utils.reward_score", score_module)

    module_path = Path(__file__).parents[3] / "verl" / "experimental" / "reward" / "reward_loop" / "naive.py"
    spec = importlib.util.spec_from_file_location("naive_reward_loop_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_reward_extra_info


def test_build_reward_extra_info_merges_agent_extra_fields(monkeypatch):
    build_reward_extra_info = load_build_reward_extra_info(monkeypatch)
    classification = {"label": "malignant", "confidence": 0.9}
    boxes = [{"bbox": [100, 100, 700, 700]}]
    trace = [{"turn": 1, "tool": "detect", "success": True}]
    non_tensor_batch = {
        "data_source": "test",
        "reward_model": {"ground_truth": {}},
        "extra_info": np.array([{"task_type": "composite"}], dtype=object),
        "classification_result": np.array([classification], dtype=object),
        "detection_boxes": np.array([boxes], dtype=object),
        "tool_trace": np.array([trace], dtype=object),
        "__num_turns__": np.array([5]),
    }

    result = build_reward_extra_info(non_tensor_batch)

    assert result["task_type"] == "composite"
    assert result["classification_result"] == classification
    assert result["detection_boxes"] == boxes
    assert result["tool_trace"] == trace
    assert result["num_turns"] == 5
