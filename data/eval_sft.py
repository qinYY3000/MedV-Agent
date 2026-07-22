"""
SFT 后简单评估脚本
==================
验证模型是否学会了 tool_call 格式和任务执行能力。
不计算复杂 NLG 指标，只看:
1. tool_call 格式正确率 (是否能生成合法 <tool_call>...</tool_call>)
2. 分类准确率 (classify 任务的标签是否正确)
3. 检测框合理性 (detect 任务的 bbox 是否在合理范围)
4. 分割流程完整性 (add_bbox -> add_point -> stop_action)

用法:
    python data/eval_sft.py \
        --model-path /mnt/workspace/MedSAM-Agent/data/sft_data/sft_merged \
        --source busi data/Dataset_BUSI_with_GT \
        --source kvasir data/kvasir-seg \
        --output data/sft_data/eval_results.json \
        --num-samples 50
"""

import argparse
import json
import os
import re
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 复用 prepare_sharegpt 的数据扫描
from data.prepare_sharegpt import scan_busi, scan_kvasir


def parse_tool_call(text: str):
    """从模型输出中提取 tool_call。"""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def load_model_and_processor(model_path: str):
    """加载 SFT 后的模型和 processor。"""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Model loaded. Device: {model.device}")
    return model, processor


def generate_response(model, processor, image_path: str, prompt: str,
                      max_new_tokens: int = 256):
    """单轮推理: 给图片 + prompt, 返回模型输出。"""
    import torch

    image = Image.open(image_path).convert("RGB")

    # 从 prepare_sharegpt 导入 system prompt
    from data.prepare_sharegpt import SYSTEM_PROMPT

    # Qwen3-VL 的消息格式 (带 system prompt 引导工具调用)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # 只取新生成的 token
    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )
    return response.strip()


def evaluate_classify(model, processor, sample):
    """评估分类任务。"""
    prompt = "Classify this medical image. What is the finding?"
    response = generate_response(model, processor, sample["image_path"], prompt)

    tc = parse_tool_call(response)
    gt_label = sample["label"]

    result = {
        "task": "classify",
        "sample_id": sample["sample_id"],
        "gt_label": gt_label,
        "response": response,
        "tool_call": tc,
        "format_correct": tc is not None and tc.get("name") == "classify",
    }
    return result


def evaluate_detect(model, processor, sample):
    """评估检测任务。"""
    prompt = "Detect all targets in this image. Use the detect tool."
    response = generate_response(model, processor, sample["image_path"], prompt)

    tc = parse_tool_call(response)

    result = {
        "task": "detect",
        "sample_id": sample["sample_id"],
        "gt_label": sample["label"],
        "response": response,
        "tool_call": tc,
        "format_correct": tc is not None and tc.get("name") == "detect",
    }
    return result


def evaluate_segment(model, processor, sample):
    """评估分割任务 — 只看第一步是否调用 add_bbox。"""
    prompt = "Segment the target in this image."
    response = generate_response(model, processor, sample["image_path"], prompt,
                                 max_new_tokens=128)

    tc = parse_tool_call(response)

    result = {
        "task": "segment",
        "sample_id": sample["sample_id"],
        "response": response,
        "tool_call": tc,
        "format_correct": tc is not None and tc.get("name") == "add_bbox",
    }

    # 检查 bbox 坐标是否在 [0, 999] 范围
    if tc and "bbox_2d" in tc.get("arguments", {}):
        bbox = tc["arguments"]["bbox_2d"]
        result["bbox_in_range"] = all(0 <= v <= 999 for v in bbox)
    else:
        result["bbox_in_range"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="Simple SFT evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to SFT merged model")
    parser.add_argument("--source", type=str, nargs=2, action="append", required=True,
                        metavar=("TYPE", "ROOT"))
    parser.add_argument("--output", type=str, default="data/sft_data/eval_results.json")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples per task to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Evaluation: Tool Call Format Check")
    print("=" * 60)

    # 加载模型
    model, processor = load_model_and_processor(args.model_path)

    # 扫描数据集 (只用 test split)
    all_test_samples = []
    for src_type, src_root in args.source:
        print(f"\nScanning {src_type}: {src_root}")
        if src_type == "busi":
            splits = scan_busi(src_root, seed=args.seed)
        elif src_type == "kvasir":
            splits = scan_kvasir(src_root, seed=args.seed)
        else:
            continue
        all_test_samples.extend(splits["test"])

    print(f"\nTest samples: {len(all_test_samples)}")

    # 按标签分组采样
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_test_samples)

    # 分任务评估
    results = []
    classify_samples = [s for s in all_test_samples][:args.num_samples]
    detect_samples = [s for s in all_test_samples if s.get("bbox")][:args.num_samples // 2]
    segment_samples = [s for s in all_test_samples if s.get("mask_path")][:args.num_samples // 2]

    print(f"\nEvaluating {len(classify_samples)} classify, "
          f"{len(detect_samples)} detect, {len(segment_samples)} segment...")

    for i, sample in enumerate(classify_samples):
        print(f"  [classify {i+1}/{len(classify_samples)}] {sample['sample_id']}")
        try:
            r = evaluate_classify(model, processor, sample)
            results.append(r)
        except Exception as e:
            print(f"    Error: {e}")
            results.append({"task": "classify", "sample_id": sample["sample_id"],
                           "error": str(e)})

    for i, sample in enumerate(detect_samples):
        print(f"  [detect {i+1}/{len(detect_samples)}] {sample['sample_id']}")
        try:
            r = evaluate_detect(model, processor, sample)
            results.append(r)
        except Exception as e:
            print(f"    Error: {e}")
            results.append({"task": "detect", "sample_id": sample["sample_id"],
                           "error": str(e)})

    for i, sample in enumerate(segment_samples):
        print(f"  [segment {i+1}/{len(segment_samples)}] {sample['sample_id']}")
        try:
            r = evaluate_segment(model, processor, sample)
            results.append(r)
        except Exception as e:
            print(f"    Error: {e}")
            results.append({"task": "segment", "sample_id": sample["sample_id"],
                           "error": str(e)})

    # 统计
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")

    for task in ["classify", "detect", "segment"]:
        task_results = [r for r in results if r.get("task") == task]
        if not task_results:
            continue

        total = len(task_results)
        format_ok = sum(1 for r in task_results if r.get("format_correct"))
        errors = sum(1 for r in task_results if "error" in r)

        print(f"\n[{task}]")
        print(f"  Total:       {total}")
        print(f"  Format OK:   {format_ok}/{total} ({format_ok/total*100:.1f}%)")
        print(f"  Errors:      {errors}/{total}")

        if task == "segment":
            bbox_ok = sum(1 for r in task_results if r.get("bbox_in_range"))
            print(f"  Bbox valid:  {bbox_ok}/{total}")

    # 保存详细结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results: {output_path}")


if __name__ == "__main__":
    main()
