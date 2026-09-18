"""
合并所有数据集的 parquet 到 combined/ 目录
=====================================================

用法:
  python data/combine_parquet.py --datasets-dir data/datasets --output data/datasets/combined
"""

import argparse
import pandas as pd
from pathlib import Path


def select_dataset_dirs(base: Path, include: list[str] | None = None) -> list[Path]:
    """选择参与合并的数据集目录，include 指定时严格按白名单处理。"""
    if include:
        directories = []
        for dataset_name in include:
            directory = base / dataset_name
            if not directory.is_dir():
                raise ValueError(f"Requested dataset directory not found: {dataset_name}")
            directories.append(directory)
        return directories

    excluded = {"combined", "abdct", "abdmr"}
    return sorted(
        (directory for directory in base.iterdir() if directory.is_dir() and directory.name not in excluded),
        key=lambda directory: directory.name,
    )


def concat_dataset_frames(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """统一可变元数据列类型后合并，避免 pyarrow 因跨数据集类型冲突失败。"""
    normalized = []
    for dataframe in dataframes:
        frame = dataframe.copy()
        if "frame_index" in frame.columns:
            frame["frame_index"] = frame["frame_index"].astype("string")
        normalized.append(frame)
    return pd.concat(normalized, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Combine all dataset parquets")
    parser.add_argument("--datasets-dir", type=str, default="data/datasets",
                        help="Directory containing per-dataset parquet subdirs")
    parser.add_argument("--output", type=str, default="data/datasets/combined",
                        help="Output directory for combined parquets")
    parser.add_argument("--include", nargs="+", default=None,
                        help="Only combine the specified dataset subdirectories, e.g. --include busi kvasir tn3k")
    args = parser.parse_args()

    base = Path(args.datasets_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    ds_dirs = select_dataset_dirs(base, include=args.include)
    print(f"Selected datasets: {[directory.name for directory in ds_dirs]}")

    for split in ["train", "val", "test"]:
        all_dfs = []
        for ds_dir in ds_dirs:
            p = ds_dir / f"{split}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                all_dfs.append(df)
                print(f"  {ds_dir.name}/{split}: {len(df)} samples")
        if all_dfs:
            combined = concat_dataset_frames(all_dfs)
            out_path = output / f"{split}.parquet"
            combined.to_parquet(out_path, index=False)
            print(f"Combined {split}: {len(combined)} samples -> {out_path}\n")


if __name__ == "__main__":
    main()
