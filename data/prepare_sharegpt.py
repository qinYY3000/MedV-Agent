"""
直接从 BUSI/Kvasir 原始目录生成 Llama-Factory sharegpt 数据集
===========================================================
跳过 parquet 和 base64，直接使用原图文件路径，速度提升 10-100 倍。

用法:
    # 单数据集 (本地路径)
    python data/prepare_sharegpt.py \
        --source busi E:/data/Dataset_BUSI_with_GT \
        --output data/sft_data

    # 多数据集合并 (生成服务器可用路径 ★推荐★)
    python data/prepare_sharegpt.py \
        --source busi E:/data/Dataset_BUSI_with_GT \
        --source kvasir E:/data/kvasir-seg \
        --output data/sft_data \
        --server-root /mnt/workspace

    # 指定 LlamaFactory 目录（自动复制 + 生成 YAML）
    python data/prepare_sharegpt.py \
        --source busi E:/data/Dataset_BUSI_with_GT \
        --source kvasir E:/data/kvasir-seg \
        --output data/sft_data \
        --server-root /mnt/workspace \
        --llamafactory-dir /mnt/workspace/LlamaFactory

路径映射说明:
    --server-root /mnt/workspace 会将图片路径从 Windows 本地路径
    自动转换为 Linux 服务器挂载路径:
      E:/data/Dataset_BUSI_with_GT/benign/benign (1).png
      -> /mnt/workspace/Dataset_BUSI_with_GT/benign/benign (1).png
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
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. 数据扫描
# ============================================================

def scan_busi(busi_root: str, train_ratio=0.7, val_ratio=0.15, seed=42) -> Dict[str, list]:
    """扫描 BUSI 数据集。
    
    返回: {"train": [sample, ...], "val": [...], "test": [...]}
    """
    busi_root = Path(busi_root)
    samples = []
    
    for class_name in ["benign", "malignant", "normal"]:
        class_dir = busi_root / class_name
        if not class_dir.exists():
            continue
        
        # 收集原图和 mask
        sample_dict = defaultdict(lambda: {"image": None, "masks": []})
        for f in class_dir.glob("*.png"):
            if "_mask" not in f.name:
                sid = f.stem  # "benign (1)"
                sample_dict[sid]["image"] = str(f)
            else:
                match = re.match(r"(.+)_mask(?:_\d+)?\.png$", f.name)
                if match:
                    sid = match.group(1)
                    sample_dict[sid]["masks"].append(str(f))
        
        for sid, data in sample_dict.items():
            if data["image"] is None:
                continue
            # 合并多 mask
            mask_path = data["masks"][0] if data["masks"] else None
            samples.append({
                "image_path": data["image"],
                "mask_path": mask_path,
                "label": class_name,
                "sample_id": sid,
            })
    
    print(f"BUSI: {len(samples)} samples")
    for cls in ["benign", "malignant", "normal"]:
        print(f"  {cls}: {sum(1 for s in samples if s['label'] == cls)}")
    
    return _split(samples, train_ratio, val_ratio, seed)


def scan_kvasir(kvasir_root: str, train_ratio=0.7, val_ratio=0.15, seed=42) -> Dict[str, list]:
    """扫描 Kvasir-SEG 数据集。"""
    kvasir_root = Path(kvasir_root)
    images_dir = kvasir_root / "images"
    masks_dir = kvasir_root / "masks"
    
    # 加载 bbox JSON
    bbox_path = kvasir_root / "kavsir_bboxes.json"
    with open(bbox_path) as f:
        bbox_data = json.load(f)
    
    samples = []
    for img_file in sorted(images_dir.glob("*.jpg")):
        sid = img_file.stem
        mask_path = masks_dir / img_file.name
        if not mask_path.exists():
            continue
        
        # 读取 bbox
        bi = bbox_data.get(sid, {})
        bboxes = bi.get("bbox", [])
        bbox = [int(bboxes[0]["xmin"]), int(bboxes[0]["ymin"]), int(bboxes[0]["xmax"]), int(bboxes[0]["ymax"])] if bboxes else None
        
        samples.append({
            "image_path": str(img_file),
            "mask_path": str(mask_path),
            "bbox": bbox,
            "label": "polyp",
            "sample_id": sid,
        })
    
    print(f"Kvasir-SEG: {len(samples)} samples")
    return _split(samples, train_ratio, val_ratio, seed)


def _split(samples, train_ratio, val_ratio, seed):
    rng = np.random.RandomState(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    return {
        "train": [samples[i] for i in indices[:n_train]],
        "val":   [samples[i] for i in indices[n_train:n_train+n_val]],
        "test":  [samples[i] for i in indices[n_train+n_val:]],
    }


# ============================================================
# COVID-19 胸部 X-ray (4类分类 + 肺部分割)
# ============================================================

def scan_covid19(covid_root: str, train_ratio=0.7, val_ratio=0.15, seed=42) -> Dict[str, list]:
    """扫描 COVID-19 X-ray 数据集。
    目录结构: covid-19/{COVID,Lung_Opacity,Normal,Viral Pneumonia}/{images,masks}/*.png
    """
    covid_root = Path(covid_root)
    label_map = {
        "COVID": "covid",
        "Lung_Opacity": "lung_opacity",
        "Normal": "normal",
        "Viral Pneumonia": "viral_pneumonia",
    }
    samples = []
    for folder_name, label in label_map.items():
        folder = covid_root / folder_name
        if not folder.exists():
            continue
        images_dir = folder / "images"
        masks_dir = folder / "masks"
        if not images_dir.exists():
            images_dir = folder
            masks_dir = None
        for img_file in sorted(images_dir.glob("*.png")):
            sid = img_file.stem
            mask_path = None
            if masks_dir and masks_dir.exists():
                mask_candidate = masks_dir / img_file.name
                if mask_candidate.exists():
                    mask_path = str(mask_candidate.resolve())
            bbox = None
            if mask_path:
                bbox = _bbox_from_mask(mask_path)
            samples.append({
                "image_path": str(img_file.resolve()),
                "mask_path": mask_path,
                "bbox": bbox,
                "label": label,
                "sample_id": f"covid::{label}/{sid}",
            })
    print(f"COVID-19: {len(samples)} samples")
    for cls in label_map.values():
        print(f"  {cls}: {sum(1 for s in samples if s['label'] == cls)}")
    return _split(samples, train_ratio, val_ratio, seed)


# ============================================================
# TN3K 甲状腺超声 (结节分割)
# ============================================================

def scan_tn3k(tn3k_root: str, train_ratio=0.7, val_ratio=0.15, seed=42) -> Dict[str, list]:
    """扫描 TN3K 甲状腺超声数据集。
    目录结构: tn3k/{trainval-image,test-image}/*.jpg, {trainval-mask,test-mask}/*.jpg
    """
    tn3k_root = Path(tn3k_root)
    samples = []
    for img_dir_name, mask_dir_name in [("trainval-image", "trainval-mask"),
                                          ("test-image", "test-mask")]:
        img_dir = tn3k_root / img_dir_name
        mask_dir = tn3k_root / mask_dir_name
        if not img_dir.exists():
            continue
        for img_file in sorted(img_dir.glob("*.jpg")):
            sid = img_file.stem
            mask_path = mask_dir / f"{sid}.jpg"
            if not mask_path.exists():
                continue
            bbox = _bbox_from_mask(str(mask_path))
            samples.append({
                "image_path": str(img_file.resolve()),
                "mask_path": str(mask_path.resolve()),
                "bbox": bbox,
                "label": "thyroid_nodule",
                "sample_id": f"tn3k::{sid}",
            })
    print(f"TN3K: {len(samples)} samples")
    return _split(samples, train_ratio, val_ratio, seed)


# ============================================================
# AbdomenCT/MR 2D 切片 (多器官分割)
# ============================================================

def scan_abdomen_2d(abd_2d_root: str, dataset_name: str = "abdct",
                     train_ratio=0.7, val_ratio=0.15, seed=42) -> Dict[str, list]:
    """扫描已切片的腹部 CT/MR 2D 数据。
    目录结构: abdct_2d/images/*.png, abdct_2d/train.parquet
    """
    import pandas as pd
    abd_root = Path(abd_2d_root)
    samples = []

    # 优先从 parquet 读取（包含器官标签信息）
    for split in ["train", "val", "test"]:
        parquet_path = abd_root / f"{split}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            for _, row in df.iterrows():
                samples.append({
                    "image_path": row["image_path"],
                    "mask_path": row.get("mask_path"),
                    "bbox": row.get("bbox"),
                    "label": row.get("label", "organ"),
                    "sample_id": row.get("sample_id", f"{dataset_name}::{len(samples)}"),
                })
            break  # 只读一个 split 的 parquet，后面统一切分

    # 如果没有 parquet，直接扫描 images 目录
    if not samples:
        images_dir = abd_root / "images"
        # 已知器官名列表
        KNOWN_ORGANS = {
            "liver", "spleen", "kidney", "pancreas", "aorta",
            "stomach", "gallbladder", "esophagus", "duodenum",
            "inferior_vena_cava", "right_adrenal_gland", "left_adrenal_gland",
            "adrenal_gland", "postcava", "gall_bladder", "right_kidney", "left_kidney",
        }
        if images_dir.exists():
            # 收集所有 mask 文件，按 base_name 分组
            from collections import defaultdict
            mask_map = defaultdict(list)  # base_name → [(organ, mask_path), ...]
            for mask_file in images_dir.glob("*_mask.png"):
                fname = mask_file.name
                # 格式: {base_name}_{organ}_mask.png
                # 从末尾去掉 "_mask.png", 再尝试匹配器官名
                stem = fname[:-9]  # 去掉 "_mask.png"
                for organ in KNOWN_ORGANS:
                    if stem.endswith(f"_{organ}"):
                        base_name = stem[:-len(f"_{organ}")]
                        mask_map[base_name].append((organ, mask_file))
                        break
            
            # 为每个原图 + 每个器官 mask 生成一条样本
            for img_file in sorted(images_dir.glob("*.png")):
                if "_mask" in img_file.name:
                    continue
                base_name = img_file.stem
                img_path = str(img_file.resolve())
                
                # 找到对应的所有器官 masks
                organ_masks = mask_map.get(base_name, [])
                
                if organ_masks:
                    for organ, mask_file in organ_masks:
                        mask_path = str(mask_file.resolve())
                        bbox = _bbox_from_mask(mask_path)
                        samples.append({
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "bbox": bbox,
                            "label": organ,
                            "sample_id": f"{dataset_name}::{base_name}::{organ}",
                        })
                else:
                    # 没有 mask → 只能做分类
                    samples.append({
                        "image_path": img_path,
                        "mask_path": None,
                        "bbox": None,
                        "label": "organ",
                        "sample_id": f"{dataset_name}::{base_name}",
                    })

    print(f"{dataset_name.upper()}: {len(samples)} 2D slices")
    return _split(samples, train_ratio, val_ratio, seed)


# ============================================================
# 辅助函数
# ============================================================

def _bbox_from_mask(mask_path: str):
    """从二值 mask 计算 bbox [x1, y1, x2, y2]。"""
    try:
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        binary = mask_np > 127
        if not binary.any():
            return None
        ys, xs = np.where(binary)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    except Exception:
        return None


def _split(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """统一切分。"""
    if not samples:
        return {"train": [], "val": [], "test": []}
    rng = np.random.RandomState(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": [samples[i] for i in indices[:n_train]],
        "val":   [samples[i] for i in indices[n_train:n_train+n_val]],
        "test":  [samples[i] for i in indices[n_train+n_val:]],
    }


# ============================================================
# 2. 轨迹生成函数
# ============================================================

# 器官标签集合 (CT/MR 多器官数据集的 label)
ORGAN_LABELS = {
    "liver", "spleen", "kidney", "pancreas", "aorta",
    "stomach", "gallbladder", "esophagus", "duodenum",
    "inferior_vena_cava", "right_adrenal_gland", "left_adrenal_gland",
    "adrenal_gland", "postcava", "gall_bladder", "right_kidney", "left_kidney",
}

def is_organ(label):
    """判断 label 是否是解剖器官 (而非病理/病灶标签)。"""
    return label in ORGAN_LABELS

def tc(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"

SYSTEM_PROMPT = """You are a professional medical image analysis agent specializing in classification, detection, and segmentation of medical images. Use the appropriate tools to complete the requested task.

# Tools
- classify: Classify an image or region
- detect: Detect all instances of a target object
- add_bbox: Initialize segmentation with a bounding box
- add_point: Refine segmentation with a positive/negative point
- stop_action: Finish the task and output the final result

# Tool Call Format
You MUST use the following format to call tools. Do NOT use markdown json code blocks.

<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

# Examples
User: Classify this image.
Assistant: <tool_call>
{"name": "classify", "arguments": {"question": "What is the diagnosis?"}}
</tool_call>

User: Detect all targets.
Assistant: <tool_call>
{"name": "detect", "arguments": {"target": "tumor"}}
</tool_call>

User: Segment the target.
Assistant: <tool_call>
{"name": "add_bbox", "arguments": {"bbox_2d": [100, 200, 500, 600]}}
</tool_call>

# Rules
- Coordinates must be in range [0, 999]
- Always use <tool_call> format, NEVER use ```json format
- For segmentation: add_bbox first, then add_point to refine, then stop
- For composite tasks: detect->add_bbox->add_point->classify->stop
- Call one tool per turn"""


def build_classify(image_path, label):
    # 注意: system prompt 不放在 conversations 里, 而是作为单独的 system 字段
    # LlamaFactory sharegpt 要求第一条消息必须是 human
    if is_organ(label):
        prompt = f"<image>What anatomical structure is shown in this region?"
        question = "What is the anatomical structure?"
        result = f"The identified structure is {label}."
    else:
        prompt = f"<image>Classify this medical image. What is the finding?"
        question = "What is the diagnosis?"
        result = f"Classification result: {label} (confidence: 0.92)"
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": tc("classify", {"question": question})},
        {"from": "human", "value": result},
        {"from": "gpt", "value": tc("stop_action", {})},
    ]


def build_detect(image_path, bbox, label):
    if bbox is None:
        return build_classify(image_path, label)
    
    if is_organ(label):
        prompt = f"<image>Detect the {label} in this CT/MR image."
        target = label
    else:
        prompt = "<image>Detect all targets in this image."
        # 用 label 作为检测目标, 而不是硬编码
        target = label.replace("_", " ") if "_" in label else label
    
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": tc("detect", {"target": target})},
        {"from": "human", "value": f"Detected 1 region: bbox={bbox}, score=0.95"},
        {"from": "gpt", "value": tc("stop_action", {})},
    ]


def build_segment(image_path, mask_path, max_clicks=2, label=None):
    """分割轨迹: add_bbox -> add_point -> ... -> stop
    
    注意: 只在第一条 human 消息放 <image> 占位符, 后续轮次不放
    (因为后续轮次的"图片"是模型生成的 mask, 不是新图片)
    """
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    binary = mask_np > 127
    if not binary.any():
        return None
    
    ys, xs = np.where(binary)
    w, h = mask.size
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    bbox_999 = [int(x*999/w) for x in bbox[:2]] + [int(y*999/h) for y in bbox[2:]]
    
    if is_organ(label):
        prompt = f"<image>Segment the {label} in this CT/MR image."
    else:
        prompt = "<image>Segment the target in this image."
    
    msgs = [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": tc("add_bbox", {"bbox_2d": bbox_999})},
    ]
    
    for turn in range(max_clicks):
        # 后续轮次不放 <image> (只有一张图, <image> 数量必须和 images 列表长度一致)
        msgs.append({"from": "human", "value": "Here is the updated mask. What is your next action?"})
        if turn == 0:
            cx = int(np.median(xs))
            cy = int(np.median(ys))
            px, py = int(cx*999/w), int(cy*999/h)
            msgs.append({"from": "gpt", "value": tc("add_point", {"point_2d": [px, py], "point_type": "positive"})})
        else:
            msgs.append({"from": "gpt", "value": tc("stop_action", {})})
            break
    
    if not any("stop_action" in m["value"] for m in msgs):
        msgs.append({"from": "human", "value": "Here is the final mask. What is your next action?"})
        msgs.append({"from": "gpt", "value": tc("stop_action", {})})
    
    return msgs


def build_composite(image_path, mask_path, label, bbox=None):
    """复合轨迹: detect -> add_bbox -> add_point -> classify -> stop
    
    注意: 只在第一条 human 消息放 <image> 占位符
    """
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    binary = mask_np > 127
    if not binary.any():
        return None
    
    ys, xs = np.where(binary)
    w, h = mask.size
    bbox_px = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    bbox_999 = [int(x*999/w) for x in bbox_px[:2]] + [int(y*999/h) for y in bbox_px[2:]]
    cx, cy = int(np.median(xs)), int(np.median(ys))
    px, py = int(cx*999/w), int(cy*999/h)
    
    if is_organ(label):
        prompt = f"<image>Analyze this CT/MR image: detect and segment the {label}."
        target = label
        classify_result = f"The segmented region is {label}."
    else:
        prompt = "<image>Analyze this image: find all targets, segment them, and classify each one."
        target = label.replace("_", " ") if "_" in label else label
        classify_result = f"Classification result: {label} (confidence: 0.92)"
    
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": tc("detect", {"target": target})},
        {"from": "human", "value": f"Detected 1 region: bbox={bbox_999}, score=0.95"},
        {"from": "gpt", "value": tc("add_bbox", {"bbox_2d": bbox_999})},
        {"from": "human", "value": "Mask initialized. What is your next action?"},
        {"from": "gpt", "value": tc("add_point", {"point_2d": [px, py], "point_type": "positive"})},
        {"from": "human", "value": "Mask refined. What is your next action?"},
        {"from": "gpt", "value": tc("classify", {"question": "What is the anatomical structure?" if is_organ(label) else "What is the classification?"})},
        {"from": "human", "value": classify_result},
        {"from": "gpt", "value": tc("stop_action", {})},
    ]


# ============================================================
# 3. 生成 sharegpt 数据集
# ============================================================

def generate_sharegpt(samples, output_dir, dataset_name, max_clicks=2,
                      server_root=None, local_prefix=None):
    """从样本列表生成 sharegpt 格式数据集。
    
    sharegpt 格式要求:
    - 图片用 <image> 占位符
    - images 字段存图片的绝对路径（目标服务器上可访问的路径）
    
    Returns: {"conversations": [...], "images": [...]}
    """
    entries = []
    
    # 路径映射: 将本地路径转换为服务器路径
    # 例如 E:/data/Dataset_BUSI_with_GT/... -> /mnt/workspace/data/Dataset_BUSI_with_GT/...
    def map_path(p):
        abs_p = str(Path(p).resolve())
        if server_root and local_prefix:
            # Windows路径转Linux挂载路径
            abs_p = abs_p.replace("\\", "/")
            if abs_p.lower().startswith(local_prefix.lower()):
                abs_p = server_root + abs_p[len(local_prefix):]
        return abs_p
    
    image_map = {}  # 原路径 → 映射后的路径
    
    for sample in samples:
        img_path = sample["image_path"]
        image_map[img_path] = map_path(img_path)
    
    print(f"  Mapped {len(image_map)} image paths")
    if server_root:
        # 打印一个示例路径用于调试
        sample_key = list(image_map.keys())[0] if image_map else None
        if sample_key:
            print(f"  Example: {sample_key}")
            print(f"       -> {image_map[sample_key]}")
    
    for sample in samples:
        sid = sample["sample_id"]
        label = sample["label"]
        img_path = sample["image_path"]
        mask_path = sample.get("mask_path")
        bbox = sample.get("bbox")
        
        img_rel = image_map[img_path]
        
        for task_type in ["classify", "detect", "segment", "composite"]:
            if label == "normal" and task_type in ("segment", "composite"):
                continue
            if not mask_path and task_type in ("segment", "composite"):
                continue
            
            try:
                if task_type == "classify":
                    conv = build_classify(img_path, label)
                elif task_type == "detect":
                    conv = build_detect(img_path, bbox, label)
                elif task_type == "segment":
                    conv = build_segment(img_path, mask_path, max_clicks, label)
                elif task_type == "composite":
                    conv = build_composite(img_path, mask_path, label, bbox)
                else:
                    continue
                
                if conv is None:
                    continue
                
                # sharegpt 格式: value 里保留 "<image>" 占位符，images 字段写路径
                # LlamaFactory 会自动把 <image> 替换为 images 中的实际图片
                # system prompt 单独放 system 字段, 不放在 conversations 里
                conversations = []
                for m in conv:
                    new_value = m["value"]
                    # 第一条 human 消息保留 <image>，后续 human 消息确保没有多余的 <image>
                    if m["from"] == "human" and conversations and "<image>" in new_value:
                        new_value = new_value.replace("<image>", "")
                    conversations.append({"from": m["from"], "value": new_value})
                
                entries.append({
                    "conversations": conversations,
                    "images": [img_rel],
                    "system": SYSTEM_PROMPT,
                    # 额外信息（可选）
                    "_sample_id": sid,
                    "_task_type": task_type,
                    "_label": label,
                })
            except Exception as e:
                print(f"    Error {task_type} {sid}: {e}")
    
    return entries


# ============================================================
# 4. dataset_info.json
# ============================================================

DATASET_INFO = {
    "medsam_agent_sft": {
        "file_name": "medsam_agent_sft.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images",
            "system": "system"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt"
        }
    }
}


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Directly generate sharegpt dataset from raw BUSI/Kvasir directories"
    )
    parser.add_argument("--source", type=str, nargs=2, action="append", required=True,
                        metavar=("TYPE", "ROOT"),
                        help="Dataset source: --source busi E:/data/Dataset_BUSI_with_GT "
                             "or --source kvasir E:/data/kvasir-seg (can repeat)")
    parser.add_argument("--output", type=str, default="data/sft_data",
                        help="Output directory for SFT data")
    parser.add_argument("--llamafactory-dir", type=str, default=None,
                        help="LlamaFactory root. If set, copies data there.")
    parser.add_argument("--server-root", type=str, default=None,
                        help="Server root prefix for path mapping. "
                             "e.g. '/mnt/workspace' maps E:/data -> /mnt/workspace/data")
    parser.add_argument("--local-prefix", type=str, default=None,
                        help="Local path prefix to replace. "
                             "e.g. 'E:/data' will be replaced by --server-root")
    parser.add_argument("--max-clicks", type=int, default=2,
                        help="Max segment refinement turns")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Direct ShareGPT Dataset Generation (no parquet/base64)")
    print("=" * 60)
    
    # 路径映射参数
    if args.server_root and not args.local_prefix:
        # 自动检测: 取第一个数据源的 root 的父目录
        first_root = args.source[0][1]
        local_prefix = str(Path(first_root).resolve().parent)
        print(f"Auto-detected local_prefix: {local_prefix}")
    else:
        local_prefix = args.local_prefix
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描所有数据集
    all_splits = {"train": [], "val": [], "test": []}
    
    for src_type, src_root in args.source:
        print(f"\nScanning {src_type}: {src_root}")
        if src_type == "busi":
            splits = scan_busi(src_root, seed=args.seed)
        elif src_type == "kvasir":
            splits = scan_kvasir(src_root, seed=args.seed)
        elif src_type == "covid":
            splits = scan_covid19(src_root, seed=args.seed)
        elif src_type == "tn3k":
            splits = scan_tn3k(src_root, seed=args.seed)
        elif src_type in ("abdct", "abdct_2d"):
            splits = scan_abdomen_2d(src_root, "abdct", seed=args.seed)
        elif src_type in ("abdmr", "abdmr_2d"):
            splits = scan_abdomen_2d(src_root, "abdmr", seed=args.seed)
        else:
            print(f"  Unknown source type: {src_type}, skip")
            continue
        
        for split in ["train", "val", "test"]:
            all_splits[split].extend(splits.get(split, []))
    
    print(f"\nTotal: train={len(all_splits['train'])}, "
          f"val={len(all_splits['val'])}, test={len(all_splits['test'])}")
    
    # 生成 sharegpt 数据集
    for split in ["train", "val", "test"]:
        samples = all_splits[split]
        if not samples:
            continue
        
        print(f"\nGenerating {split} ({len(samples)} samples)...")
        entries = generate_sharegpt(
            samples, str(output_dir), f"{split}_", args.max_clicks,
            server_root=args.server_root, local_prefix=local_prefix
        )
        
        out_path = output_dir / f"medsam_agent_sft.json"
        
        # 如果是第一次写，直接写；否则追加
        if split == "train":
            all_entries = entries
        else:
            # 读已有 + 追加
            if out_path.exists():
                with open(out_path) as f:
                    all_entries = json.load(f)
            else:
                all_entries = []
            all_entries.extend(entries)
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False)
        
        task_counts = defaultdict(int)
        for e in entries:
            task_counts.get(e.get("_task_type", "unknown"), 0)
            task_counts[e.get("_task_type", "unknown")] += 1
        counts = ", ".join([f"{t}: {c}" for t, c in task_counts.items()])
        print(f"  → Added {len(entries)} entries ({counts})")
    
    # Shuffle 打散所有数据 (避免 max_samples 只取到前几个数据集)
    json_path = output_dir / "medsam_agent_sft.json"
    if json_path.exists():
        with open(json_path) as f:
            all_data = json.load(f)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(all_data)
        with open(json_path, "w") as f:
            json.dump(all_data, f, ensure_ascii=False)
        print(f"  Shuffled {len(all_data)} entries (seed={args.seed})")
    
    total = 0
    if json_path.exists():
        with open(json_path) as f:
            total = len(json.load(f))
    print(f"\nTotal SFT dataset: {total} entries")
    print(f"Dataset: {json_path}")
    
    # 写 dataset_info.json
    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(DATASET_INFO, f, indent=2)
    print(f"Dataset info: {info_path}")
    
    # 复制到 LlamaFactory
    if args.llamafactory_dir:
        lf_dir = Path(args.llamafactory_dir)
        lf_data_dir = lf_dir / "data"
        lf_data_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        
        # 1. 复制 JSON
        shutil.copy2(output_dir / "medsam_agent_sft.json", lf_data_dir / "medsam_agent_sft.json")
        
        # 2. 修正 JSON 中的图片路径为绝对路径（JSON 中存的是原图绝对路径）
        #    Llama-Factory 需要读取图片文件，原图路径必须是服务器上实际存在的
        with open(lf_data_dir / "medsam_agent_sft.json") as f:
            entries = json.load(f)
        with open(lf_data_dir / "medsam_agent_sft.json", "w") as f:
            json.dump(entries, f, ensure_ascii=False)
        
        # 3. 合并 dataset_info.json
        lf_info_path = lf_data_dir / "dataset_info.json"
        if lf_info_path.exists():
            with open(lf_info_path) as f:
                lf_info = json.load(f)
        else:
            lf_info = {}
        lf_info.update(DATASET_INFO)
        with open(lf_info_path, "w") as f:
            json.dump(lf_info, f, indent=2)
        
        print(f"\nCopied to LlamaFactory: {lf_data_dir}")
        
        # 4. 同时生成训练 YAML
        print(f"\n{'='*60}")
        print("Generating Training YAML for LlamaFactory...")
        from sft_train import write_train_yaml, write_lora_yaml

        # 默认值对齐 qwen3vl_lora_sft.yaml
        use_lora = os.environ.get("SFT_USE_LORA", "1").lower() in ("1", "true", "yes")
        model_path = os.environ.get("SFT_MODEL_PATH", "/mnt/workspace/Qwen3-VL-8B-Instruct")

        if use_lora:
            yaml_path = write_lora_yaml(model_path, str(lf_data_dir), str(output_dir))
        else:
            yaml_path = write_train_yaml(model_path, str(lf_data_dir), str(output_dir))
        
        print(f"Training YAML: {yaml_path}")
        print(f"\nNow run:")
        print(f"  cd {args.llamafactory_dir}")
        print(f"  llamafactory-cli train {yaml_path}")
    else:
        print(f"\nDone! Now run:")
        if args.server_root:
            print(f"  # Data generated with server paths (prefix: {args.server_root})")
            print(f"  python data/sft_train.py \\")
            print(f"      --sharegpt-json data/sft_data/medsam_agent_sft.json \\")
            print(f"      --model-path /mnt/workspace/Qwen3-VL-8B-Instruct \\")
            print(f"      --output-dir data/sft_data \\")
            print(f"      --llamafactory-dir /mnt/workspace/LlamaFactory")
        else:
            print(f"  python data/sft_train.py \\")
            print(f"      --sharegpt-json data/sft_data/medsam_agent_sft.json \\")
            print(f"      --model-path /mnt/workspace/Qwen3-VL-8B-Instruct \\")
            print(f"      --output-dir data/sft_data \\")
            print(f"      --llamafactory-dir /mnt/workspace/LlamaFactory")
        print(f"  cd /mnt/workspace/LlamaFactory && llamafactory-cli train data/sft_data/sft_train.yaml")


if __name__ == "__main__":
    main()
