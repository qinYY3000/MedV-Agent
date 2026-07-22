"""
从 combined parquet 中采样子集
==============================
每个数据集均匀采样，保证模态平衡。

用法:
  python data/sample_subset.py \
    --input-dir data/datasets/combined \
    --output-dir data/datasets/subset \
    --total-samples 2000
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Sample balanced subset from combined parquet")
    parser.add_argument("--input-dir", type=str, default="data/datasets/combined")
    parser.add_argument("--output-dir", type=str, default="data/datasets/subset")
    parser.add_argument("--total-samples", type=int, default=2000,
                        help="Total samples to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        parquet_path = input_dir / f"{split}.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        print(f"\n{split}: {len(df)} total samples")

        # 按数据集分组 (从 sample_id 提取前缀)
        df["dataset"] = df["sample_id"].apply(
            lambda x: x.split("::")[0] if "::" in str(x) else x.split("__")[0] if "__" in str(x) else "unknown"
        )
        dataset_counts = df["dataset"].value_counts()
        print(f"  Datasets: {dict(dataset_counts)}")

        if split == "train":
            # 训练集: 按比例采样
            total = args.total_samples
        elif split == "val":
            total = max(100, args.total_samples // 10)
        else:
            total = max(100, args.total_samples // 10)

        # 每个数据集采样数量 (按比例分配, 但至少 10 个)
        n_datasets = len(dataset_counts)
        per_dataset = max(10, total // n_datasets)

        sampled_dfs = []
        rng = np.random.RandomState(args.seed)

        for ds_name in dataset_counts.index:
            ds_df = df[df["dataset"] == ds_name]
            n = min(per_dataset, len(ds_df))
            if n < len(ds_df):
                indices = rng.choice(len(ds_df), size=n, replace=False)
                ds_sampled = ds_df.iloc[indices]
            else:
                ds_sampled = ds_df
            sampled_dfs.append(ds_sampled)
            print(f"    {ds_name}: {len(ds_sampled)}/{len(ds_df)}")

        result = pd.concat(sampled_dfs, ignore_index=True)
        result = result.drop(columns=["dataset"])

        out_path = output_dir / f"{split}.parquet"
        result.to_parquet(out_path, index=False)
        print(f"  Saved {len(result)} samples -> {out_path}")

    print(f"\n{'='*60}")
    print(f"Subset created at: {output_dir}")
    print(f"Use in run_multi_task.sh:")
    print(f"  DATASET_TRAIN=data/datasets/subset/train.parquet")
    print(f"  DATASET_VAL=data/datasets/subset/val.parquet")


if __name__ == "__main__":
    main()
