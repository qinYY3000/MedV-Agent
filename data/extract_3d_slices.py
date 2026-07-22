"""
3D 医学体数据 → 2D 切片提取
====================================
AbdomenCT/MR 的 NIfTI 文件需要切为 2D 图像才能用于 VLM 训练。

策略:
  - 读取 3D volume 和 label
  - 选择有目标的中间切片 (label 非空)
  - 每个体数据选 3-5 张有代表性的切片
  - 保存为 PNG (image + mask)

用法:
  python data/extract_3d_slices.py \
    --input-dir data/datasets/abdct \
    --output-dir data/datasets/abdct_2d \
    --slices-per-volume 5
"""

import argparse
import json
import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def load_nifti(path: str):
    """加载 NIfTI 文件，返回 numpy array。"""
    try:
        import nibabel as nib
        img = nib.load(path)
        return img.get_fdata()
    except ImportError:
        # 尝试 SimpleITK
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(img)
        except ImportError:
            raise ImportError(
                "需要 nibabel 或 SimpleITK 来读取 NIfTI 文件。\n"
                "安装: pip install nibabel 或 pip install SimpleITK"
            )


def normalize_slice(slice_2d: np.ndarray, modality: str = "ct") -> np.ndarray:
    """归一化 2D 切片到 0-255。"""
    if modality == "ct":
        # CT: 窗宽窗位 (abdomen: W=400, L=40)
        window, level = 400, 40
        lower = level - window / 2
        upper = level + window / 2
        slice_2d = np.clip(slice_2d, lower, upper)
        slice_2d = (slice_2d - lower) / (upper - lower) * 255
    else:
        # MR: percentile 归一化
        p1, p99 = np.percentile(slice_2d, [1, 99])
        if p99 > p1:
            slice_2d = np.clip(slice_2d, p1, p99)
            slice_2d = (slice_2d - p1) / (p99 - p1) * 255
        else:
            slice_2d = np.zeros_like(slice_2d)
    return slice_2d.astype(np.uint8)


def find_best_slices(label_vol: np.ndarray, n_slices: int = 5) -> Tuple[List[int], int]:
    """找到有最多目标像素的 n 个切片。

    Returns:
        (selected_slice_indices, slice_axis)
        slice_axis: 0, 1, or 2 — 切片所在的维度
    """
    if label_vol.ndim != 3:
        return list(range(min(n_slices, label_vol.shape[0]))), 0

    # 医学图像 NIfTI 通常是 (H, W, D) 或 (W, H, D)
    # 切片维度是最小的那个维度 (通常是 D)
    shape = label_vol.shape
    slice_axis = min(range(3), key=lambda i: shape[i])

    # 找到每个切片的非零像素数
    slice_scores = []
    for i in range(shape[slice_axis]):
        # 取第 slice_axis 维的第 i 个切片
        slc = np.take(label_vol, i, axis=slice_axis)
        score = np.sum(slc > 0)
        slice_scores.append((i, score))

    # 按分数排序，取 top n，再按位置排序
    slice_scores.sort(key=lambda x: -x[1])
    selected = sorted([s[0] for s in slice_scores[:n_slices]])
    return selected, slice_axis


def extract_slices_from_volume(
    image_path: str,
    label_path: str,
    output_dir: str,
    sample_id: str,
    modality: str = "ct",
    label_names: dict = None,
    n_slices: int = 5,
) -> List[dict]:
    """从 3D 体数据提取 2D 切片。"""
    image_vol = load_nifti(image_path)
    label_vol = load_nifti(label_path)

    # 确保维度一致
    if image_vol.shape != label_vol.shape:
        # 尝试转置
        if image_vol.shape[::-1] == label_vol.shape:
            label_vol = np.transpose(label_vol, (2, 1, 0))
        elif image_vol.shape == label_vol.shape[::-1]:
            label_vol = np.transpose(label_vol, (2, 1, 0))

    # 找最佳切片 (自动检测切片维度)
    best_slices, slice_axis = find_best_slices(label_vol, n_slices)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建器官标签映射 (只保留有效器官，排除 background)
    # label_names 格式: {"liver": 1, "spleen": 3, "background": 0, ...}
    # 需要翻转为 {1: "liver", 3: "spleen", ...}
    organ_labels = {}
    if label_names:
        for lbl_name, lbl_id in label_names.items():
            try:
                lbl_id_int = int(lbl_id)
            except (ValueError, TypeError):
                continue
            if lbl_id_int > 0:
                organ_labels[lbl_id_int] = lbl_name
    if not organ_labels:
        # fallback: 如果没有 label_names，用二值 mask
        organ_labels = {1: "organ"}

    samples = []
    for slice_idx in best_slices:
        # 沿正确的轴取切片
        img_slice = np.take(image_vol, slice_idx, axis=slice_axis)
        label_slice = np.take(label_vol, slice_idx, axis=slice_axis)

        # 跳过空切片
        if np.sum(label_slice > 0) == 0:
            continue

        # 归一化图像
        img_norm = normalize_slice(img_slice, modality)

        # 保存原图 (每张切片只保存一次)
        safe_id = sample_id.replace("::", "__").replace(":", "_")
        sid = f"{safe_id}_slice{slice_idx:03d}"
        img_path = output_dir / f"{sid}.png"
        Image.fromarray(img_norm).save(str(img_path))

        # 为每个器官生成独立的 mask
        for organ_id, organ_name in organ_labels.items():
            organ_mask = (label_slice == organ_id).astype(np.uint8) * 255
            if organ_mask.sum() == 0:
                continue  # 该切片没有这个器官

            # 保存器官 mask
            organ_safe = organ_name.replace(" ", "_")
            mask_path = output_dir / f"{sid}_{organ_safe}_mask.png"
            Image.fromarray(organ_mask).save(str(mask_path))

            # 计算 bbox
            ys, xs = np.where(organ_mask > 0)
            if len(ys) == 0:
                continue
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            samples.append({
                "image_path": str(img_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "bbox": bbox,
                "label": organ_name,
                "modality": modality,
                "anatomy": "abdomen",
                "sample_id": f"{safe_id}__slice{slice_idx:03d}__{organ_safe}",
                "slice_idx": slice_idx,
                "organ_id": organ_id,
                "is_3d": False,
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract 2D slices from 3D volumes")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Input parquet directory (abdct or abdmr). Required for full extraction, not needed for --parquet-only.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for 2D images + parquet")
    parser.add_argument("--slices-per-volume", type=int, default=5,
                        help="Number of slices to extract per volume")
    parser.add_argument("--parquet-only", action="store_true",
                        help="Only regenerate parquet from existing PNG slices (no re-slicing). "
                             "Use this when images are already on server but parquet paths need fixing.")
    args = parser.parse_args()

    import pandas as pd

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 模式1: parquet-only (不重新切片，只从已有 PNG 重建 parquet) ----
    if args.parquet_only:
        print("=" * 60)
        print("Parquet-only mode: rebuild parquet from existing PNG slices")
        print("=" * 60)

        images_dir = output_dir / "images"
        if not images_dir.exists():
            print(f"Error: {images_dir} not found. Run full extraction first.")
            return

        # 收集所有切片 PNG
        # 图片文件名: abdmr__amos_0507_0000_slice048.png
        # mask 文件名: abdmr__amos_0507_0000_slice048_liver_mask.png
        all_samples = []
        for img_file in sorted(images_dir.glob("*.png")):
            if "_mask" in img_file.name:
                continue  # 跳过 mask 文件，只处理图片

            stem = img_file.stem  # abdmr__amos_0507_0000_slice048

            # 解析数据集前缀和 modality
            ds_prefix = stem.split("__")[0]  # abdmr 或 abdct
            modality = "ct" if ds_prefix == "abdct" else "mr"

            # 提取 slice_idx
            import re
            slice_match = re.search(r"slice(\d+)", stem)
            if not slice_match:
                continue
            slice_idx = int(slice_match.group(1))

            # 查找该图片对应的所有 organ mask
            # mask 命名: {stem}_{organ}_mask.png
            organ_masks = list(images_dir.glob(f"{stem}_*_mask.png"))
            if not organ_masks:
                continue

            for mask_file in organ_masks:
                # 从 mask 文件名提取 organ 名
                # abdmr__amos_0507_0000_slice048_liver_mask.png → liver
                mask_stem = mask_file.stem  # abdmr__amos_0507_0000_slice048_liver_mask
                organ_name = mask_stem.replace(stem + "_", "").replace("_mask", "")
                # 还原空格: right_kidney → right kidney
                organ_display = organ_name.replace("_", " ")

                # 从 mask 计算 bbox
                bbox = _bbox_from_mask(str(mask_file))

                all_samples.append({
                    "image_path": str(img_file.resolve()),
                    "mask_path": str(mask_file.resolve()),
                    "bbox": bbox,
                    "label": organ_display,
                    "modality": modality,
                    "anatomy": "abdomen",
                    "sample_id": f"{stem}__{organ_name}",
                    "slice_idx": slice_idx,
                    "organ_id": 0,
                    "is_3d": False,
                })

        # 按样本 ID 重新切分 train/val/test
        rng = np.random.RandomState(42)
        indices = list(range(len(all_samples)))
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        splits = {
            "train": [all_samples[i] for i in indices[:n_train]],
            "val":   [all_samples[i] for i in indices[n_train:n_train + n_val]],
            "test":  [all_samples[i] for i in indices[n_train + n_val:]],
        }

        for split_name, split_samples in splits.items():
            if split_samples:
                out_path = output_dir / f"{split_name}.parquet"
                pd.DataFrame(split_samples).to_parquet(out_path, index=False)
                print(f"  {split_name}: {len(split_samples)} samples -> {out_path}")

        print(f"\nTotal: {len(all_samples)} samples")
        return

    # ---- 模式2: 完整切片 (读取 3D NIfTI → 2D PNG + parquet) ----
    if not args.input_dir:
        print("Error: --input-dir is required for full extraction mode")
        return
    input_dir = Path(args.input_dir)
    for split in ["train", "val", "test"]:
        parquet_path = input_dir / f"{split}.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        print(f"\nProcessing {split}: {len(df)} volumes")

        all_samples = []
        for idx, row in df.iterrows():
            if not row.get("is_3d", False):
                continue
            try:
                modality = row["modality"]
                label_names = row.get("label_names", {})
                if isinstance(label_names, str):
                    try:
                        label_names = json.loads(label_names)
                    except json.JSONDecodeError:
                        label_names = {}
                elif label_names is None or (hasattr(label_names, '__iter__') and not isinstance(label_names, dict)):
                    label_names = {}
                if not label_names:
                    ds_json = Path(row["image_path"]).parent.parent / "dataset.json"
                    if ds_json.exists():
                        with open(ds_json) as f:
                            cfg = json.load(f)
                        label_names = cfg.get("labels", {})
                samples = extract_slices_from_volume(
                    image_path=row["image_path"],
                    label_path=row["mask_path"],
                    output_dir=str(output_dir / "images"),
                    sample_id=row["sample_id"],
                    modality=modality,
                    label_names=label_names if isinstance(label_names, dict) else None,
                    n_slices=args.slices_per_volume,
                )
                all_samples.extend(samples)
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(df)} volumes, {len(all_samples)} slices")
            except Exception as e:
                print(f"  Error {row['sample_id']}: {e}")

        if all_samples:
            out_parquet = output_dir / f"{split}.parquet"
            pd.DataFrame(all_samples).to_parquet(out_parquet, index=False)
            print(f"  Saved {len(all_samples)} 2D slices -> {out_parquet}")


if __name__ == "__main__":
    main()
