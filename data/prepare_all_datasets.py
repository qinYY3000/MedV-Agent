"""
统一多模态医学数据集预处理
====================================
支持 6 个数据集:
  1. BUSI          - 乳腺超声 (benign/malignant/normal)
  2. Kvasir-SEG    - 息肉内镜
  3. COVID-19      - 胸部 X-ray (covid/lung_opacity/normal/pneumonia)
  4. TN3K          - 甲状腺超声结节
  5. AbdomenCT     - 腹部 CT 多器官分割 (liver/kidney/spleen/pancreas...)
  6. AbdomenMR     - 腹部 MR 多器官分割

输出:
  - 统一 parquet 格式 (train/val/test split)
  - 每条记录: {image_path, mask_path, bbox, instances, label, modality, anatomy, sample_id}
  - `mask_path` / `bbox` 保留首个实例以兼容单目标训练；`instances` 保留同图全部实例
  - 按统一 7:1.5:1.5 切分 (seed=42)

用法:
  python data/prepare_all_datasets.py --data-root E:/data --output data/datasets
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
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 工具函数
# ============================================================

def bbox_from_mask(mask_path: str) -> Optional[List[int]]:
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


def build_mask_instances(mask_paths: List[str], label: str, sample_id: str) -> List[dict]:
    """将同图的多个实例 mask 转换为可序列化的实例列表。"""
    instances = []
    for index, mask_path in enumerate(sorted(mask_paths)):
        resolved_path = str(Path(mask_path).resolve())
        instances.append({
            "instance_id": f"{sample_id}::{index}",
            "label": label,
            "mask_path": resolved_path,
            "bbox": bbox_from_mask(resolved_path),
            "mask_scope": "instance",
        })
    return instances


def build_bbox_instances(bboxes: List[dict], label: str, sample_id: str, mask_path: str) -> List[dict]:
    """将检测标注框转换为实例列表。

    Kvasir 当前提供图级语义 mask；每个实例保留独立 bbox，mask_scope
    显式标为 semantic_union，避免将图级 mask 误当成实例级分割真值。
    """
    resolved_mask_path = str(Path(mask_path).resolve())
    instances = []
    for index, box in enumerate(bboxes):
        instances.append({
            "instance_id": f"{sample_id}::{index}",
            "label": label,
            "mask_path": resolved_mask_path,
            "bbox": [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])],
            "mask_scope": "semantic_union",
        })
    return instances


def split_samples(samples: list, train_ratio=0.7, val_ratio=0.15, seed=42):
    """统一切分: train 70%, val 15%, test 15%。"""
    rng = np.random.RandomState(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": [samples[i] for i in indices[:n_train]],
        "val": [samples[i] for i in indices[n_train:n_train + n_val]],
        "test": [samples[i] for i in indices[n_train + n_val:]],
    }


def save_parquet(samples: list, output_path: str):
    """保存为 parquet 格式。"""
    import pandas as pd
    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(samples)} samples -> {output_path}")


# ============================================================
# 1. BUSI 乳腺超声
# ============================================================

def scan_busi(busi_root: str) -> Dict[str, list]:
    busi_root = Path(busi_root)
    samples = []
    for class_name in ["benign", "malignant", "normal"]:
        class_dir = busi_root / class_name
        if not class_dir.exists():
            continue
        sample_dict = defaultdict(lambda: {"image": None, "masks": []})
        for f in class_dir.glob("*.png"):
            if "_mask" not in f.name:
                sid = f.stem
                sample_dict[sid]["image"] = str(f)
            else:
                match = re.match(r"(.+)_mask(?:_\d+)?\.png$", f.name)
                if match:
                    sid = match.group(1)
                    sample_dict[sid]["masks"].append(str(f))
        for sid, data in sample_dict.items():
            if data["image"] is None:
                continue
            sample_id = f"busi::{sid}"
            instances = build_mask_instances(data["masks"], class_name, sample_id)
            primary_instance = instances[0] if instances else None
            samples.append({
                "image_path": str(Path(data["image"]).resolve()),
                # 保留第一实例字段，兼容当前单目标训练与奖励链路。
                "mask_path": primary_instance["mask_path"] if primary_instance else None,
                "bbox": primary_instance["bbox"] if primary_instance else None,
                "instances": instances,
                "label": class_name,
                "modality": "ultrasound",
                "anatomy": "breast",
                "sample_id": sample_id,
            })
    print(f"BUSI: {len(samples)} samples")
    for cls in ["benign", "malignant", "normal"]:
        print(f"  {cls}: {sum(1 for s in samples if s['label'] == cls)}")
    return split_samples(samples)


# ============================================================
# 2. Kvasir-SEG 息肉内镜
# ============================================================

def scan_kvasir(kvasir_root: str) -> Dict[str, list]:
    kvasir_root = Path(kvasir_root)
    images_dir = kvasir_root / "images"
    masks_dir = kvasir_root / "masks"
    bbox_path = kvasir_root / "kavsir_bboxes.json"
    with open(bbox_path) as f:
        bbox_data = json.load(f)
    samples = []
    for img_file in sorted(images_dir.glob("*.jpg")):
        sid = img_file.stem
        mask_path = masks_dir / img_file.name
        if not mask_path.exists():
            continue
        bi = bbox_data.get(sid, {})
        bboxes = bi.get("bbox", [])
        sample_id = f"kvasir::{sid}"
        instances = build_bbox_instances(bboxes, "polyp", sample_id, str(mask_path))
        primary_instance = instances[0] if instances else None
        samples.append({
            "image_path": str(img_file.resolve()),
            # 图级 mask 仍用于当前兼容路径；多实例训练应使用 instances 中的 bbox。
            "mask_path": str(mask_path.resolve()),
            "bbox": primary_instance["bbox"] if primary_instance else None,
            "instances": instances,
            "label": "polyp",
            "modality": "endoscopy",
            "anatomy": "colon",
            "sample_id": sample_id,
        })
    print(f"Kvasir-SEG: {len(samples)} samples")
    return split_samples(samples)


# ============================================================
# 3. COVID-19 胸部 X-ray
# ============================================================

def scan_covid19(covid_root: str) -> Dict[str, list]:
    covid_root = Path(covid_root)
    samples = []
    label_map = {
        "COVID": "covid",
        "Lung_Opacity": "lung_opacity",
        "Normal": "normal",
        "Viral Pneumonia": "viral_pneumonia",
    }
    for folder_name, label in label_map.items():
        folder = covid_root / folder_name
        if not folder.exists():
            continue
        images_dir = folder / "images"
        masks_dir = folder / "masks"
        if not images_dir.exists():
            # 兼容: 如果没有 images/ 子目录，直接在 folder 下找
            images_dir = folder
            masks_dir = None
        for img_file in sorted(images_dir.glob("*.png")):
            sid = img_file.stem
            # 查找对应的 mask
            mask_path = None
            if masks_dir and masks_dir.exists():
                mask_candidate = masks_dir / img_file.name
                if mask_candidate.exists():
                    mask_path = str(mask_candidate.resolve())
            # 从 mask 计算 bbox (非 Normal 类别才有意义)
            bbox = None
            if mask_path:
                bbox = bbox_from_mask(mask_path)
            samples.append({
                "image_path": str(img_file.resolve()),
                "mask_path": mask_path,
                "bbox": bbox,
                "label": label,
                "modality": "xray",
                "anatomy": "chest",
                "sample_id": f"covid::{label}/{sid}",
            })
    print(f"COVID-19: {len(samples)} samples")
    for cls in label_map.values():
        print(f"  {cls}: {sum(1 for s in samples if s['label'] == cls)}")
    return split_samples(samples)


# ============================================================
# 4. TN3K 甲状腺超声
# ============================================================

def scan_tn3k(tn3k_root: str) -> Dict[str, list]:
    tn3k_root = Path(tn3k_root)
    trainval_images = tn3k_root / "trainval-image"
    trainval_masks = tn3k_root / "trainval-mask"
    test_images = tn3k_root / "test-image"
    test_masks = tn3k_root / "test-mask"
    samples = []
    # 合并 trainval + test，统一重新切分
    for images_dir, masks_dir in [(trainval_images, trainval_masks),
                                   (test_images, test_masks)]:
        if not images_dir.exists():
            continue
        for img_file in sorted(images_dir.glob("*.jpg")):
            sid = img_file.stem
            mask_path = masks_dir / f"{sid}.jpg"
            if not mask_path.exists():
                continue
            bbox = bbox_from_mask(str(mask_path))
            samples.append({
                "image_path": str(img_file.resolve()),
                "mask_path": str(mask_path.resolve()),
                "bbox": bbox,
                "label": "thyroid_nodule",
                "modality": "ultrasound",
                "anatomy": "thyroid",
                "sample_id": f"tn3k::{sid}",
            })
    print(f"TN3K: {len(samples)} samples")
    return split_samples(samples)


# ============================================================
# 5. AbdomenCT 腹部 CT 多器官
# ============================================================

def scan_abdomen_ct(ct_root: str) -> Dict[str, list]:
    """AbdomenCT: NIfTI 3D 体数据，需要切 2D 切片。"""
    ct_root = Path(ct_root)
    dataset_json = ct_root / "dataset.json"
    with open(dataset_json) as f:
        cfg = json.load(f)
    label_names = cfg.get("labels", {})

    samples = []
    # 合并 Tr + Val，统一切分
    for split_name in ["imagesTr", "imagesVal"]:
        images_dir = ct_root / split_name
        labels_dir = ct_root / split_name.replace("images", "labels")
        if not images_dir.exists():
            continue
        for img_file in sorted(images_dir.glob("*.nii.gz")):
            sid = img_file.stem.replace(".nii", "")
            # label 文件名: 去掉 _0000 后缀
            # image: FLARE22_Tr_0001_0000.nii.gz
            # label: FLARE22_Tr_0001.nii.gz
            label_name = img_file.name.replace("_0000.nii.gz", ".nii.gz")
            label_file = labels_dir / label_name
            if not label_file.exists():
                # fallback: 尝试直接同名
                label_file = labels_dir / img_file.name
            if not label_file.exists():
                continue
            # 标记为 3D 数据，后续切片处理
            samples.append({
                "image_path": str(img_file.resolve()),
                "mask_path": str(label_file.resolve()),
                "bbox": None,
                "label": "abdomen_organ",
                "modality": "ct",
                "anatomy": "abdomen",
                "sample_id": f"abdct::{sid}",
                "label_names": label_names,
                "is_3d": True,
            })
    print(f"AbdomenCT: {len(samples)} 3D volumes")
    return split_samples(samples)


# ============================================================
# 6. AbdomenMR 腹部 MR 多器官
# ============================================================

def scan_abdomen_mr(mr_root: str) -> Dict[str, list]:
    """AbdomenMR: NIfTI 3D 体数据。"""
    mr_root = Path(mr_root)
    dataset_json = mr_root / "dataset.json"
    with open(dataset_json) as f:
        cfg = json.load(f)
    label_names = cfg.get("labels", {})

    samples = []
    for split_name in ["imagesTr", "imagesTs"]:
        images_dir = mr_root / split_name
        labels_dir = mr_root / split_name.replace("images", "labels")
        if not images_dir.exists():
            continue
        for img_file in sorted(images_dir.glob("*.nii.gz")):
            sid = img_file.stem.replace(".nii", "")
            # label 文件名: 去掉 _0000 后缀
            label_name = img_file.name.replace("_0000.nii.gz", ".nii.gz")
            label_file = labels_dir / label_name
            if not label_file.exists():
                label_file = labels_dir / img_file.name
            if not label_file.exists():
                continue
            samples.append({
                "image_path": str(img_file.resolve()),
                "mask_path": str(label_file.resolve()),
                "bbox": None,
                "label": "abdomen_organ",
                "modality": "mr",
                "anatomy": "abdomen",
                "sample_id": f"abdmr::{sid}",
                "label_names": label_names,
                "is_3d": True,
            })
    print(f"AbdomenMR: {len(samples)} 3D volumes")
    return split_samples(samples)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare all medical datasets (RL parquet format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 处理所有 6 个数据集
  python data/prepare_all_datasets.py \\
    --busi  data/Dataset_BUSI_with_GT \\
    --kvasir data/kvasir-seg \\
    --covid data/covid-19 \\
    --tn3k  data/tn3k \\
    --abdct data/Dataset701_AbdomenCT \\
    --abdmr data/Dataset702_AbdomenMR \\
    --output data/datasets

  # 只处理部分数据集
  python data/prepare_all_datasets.py \\
    --busi  data/Dataset_BUSI_with_GT \\
    --kvasir data/kvasir-seg \\
    --output data/datasets
        """
    )
    parser.add_argument("--busi", type=str, default=None,
                        help="BUSI 乳腺超声目录 (Dataset_BUSI_with_GT)")
    parser.add_argument("--kvasir", type=str, default=None,
                        help="Kvasir-SEG 息肉内镜目录")
    parser.add_argument("--covid", type=str, default=None,
                        help="COVID-19 胸部 X-ray 目录")
    parser.add_argument("--tn3k", type=str, default=None,
                        help="TN3K 甲状腺超声目录")
    parser.add_argument("--abdct", type=str, default=None,
                        help="AbdomenCT 目录 (Dataset701_AbdomenCT, 含 NIfTI)")
    parser.add_argument("--abdmr", type=str, default=None,
                        help="AbdomenMR 目录 (Dataset702_AbdomenMR, 含 NIfTI)")
    parser.add_argument("--output", type=str, default="data/datasets",
                        help="Output directory for parquet files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Unified Multi-Modal Medical Dataset Preparation (RL parquet)")
    print("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据集 → (路径参数, 扫描函数) 映射
    dataset_configs = [
        ("busi",   args.busi,   scan_busi),
        ("kvasir", args.kvasir, scan_kvasir),
        ("covid",  args.covid,  scan_covid19),
        ("tn3k",   args.tn3k,   scan_tn3k),
        ("abdct",  args.abdct,  scan_abdomen_ct),
        ("abdmr",  args.abdmr,  scan_abdomen_mr),
    ]

    all_splits = {"train": [], "val": [], "test": []}
    dataset_stats = {}

    for ds_name, ds_path, scanner in dataset_configs:
        if not ds_path:
            print(f"\n--- {ds_name}: SKIP (no path provided) ---")
            continue
        if not Path(ds_path).exists():
            print(f"\n--- {ds_name}: SKIP (path not found: {ds_path}) ---")
            continue

        print(f"\n--- {ds_name}: {ds_path} ---")
        splits = scanner(ds_path)

        # 每个数据集单独存 parquet
        ds_dir = output_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            if splits[split]:
                save_parquet(splits[split], str(ds_dir / f"{split}.parquet"))
                all_splits[split].extend(splits[split])

        dataset_stats[ds_name] = {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": sum(len(v) for v in splits.values()),
        }

    # 合并所有数据集
    print(f"\n{'='*60}")
    print("Combined Statistics")
    print(f"{'='*60}")
    for ds_name, stats in dataset_stats.items():
        print(f"  {ds_name:10s}: train={stats['train']:5d}, val={stats['val']:4d}, test={stats['test']:4d}, total={stats['total']}")
    print(f"  {'TOTAL':10s}: train={len(all_splits['train']):5d}, val={len(all_splits['val']):4d}, test={len(all_splits['test']):4d}")

    # 保存合并的 parquet
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        if all_splits[split]:
            save_parquet(all_splits[split], str(combined_dir / f"{split}.parquet"))

    # 保存统计信息
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")

    # 保存 README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Unified Medical Datasets\n\n")
        f.write("| Dataset | Modality | Anatomy | Labels | Train | Val | Test | Total |\n")
        f.write("|---------|----------|---------|--------|-------|-----|------|-------|\n")
        ds_info = {
            "busi": ("ultrasound", "breast", "benign/malignant/normal"),
            "kvasir": ("endoscopy", "colon", "polyp"),
            "covid": ("xray", "chest", "covid/lung_opacity/normal/pneumonia"),
            "tn3k": ("ultrasound", "thyroid", "thyroid_nodule"),
            "abdct": ("ct", "abdomen", "multi-organ (liver/kidney/spleen/pancreas...)"),
            "abdmr": ("mr", "abdomen", "multi-organ"),
        }
        for ds_name, stats in dataset_stats.items():
            mod, ana, lbl = ds_info.get(ds_name, ("?", "?", "?"))
            f.write(f"| {ds_name} | {mod} | {ana} | {lbl} | {stats['train']} | {stats['val']} | {stats['test']} | {stats['total']} |\n")
    print(f"README saved to: {readme_path}")


if __name__ == "__main__":
    main()
