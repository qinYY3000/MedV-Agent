"""
合并所有数据集的 parquet 到 combined/ 目录
=====================================================

用法:
  python data/combine_parquet.py --datasets-dir data/datasets --output data/datasets/combined
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Combine all dataset parquets")
    parser.add_argument("--datasets-dir", type=str, default="data/datasets",
                        help="Directory containing per-dataset parquet subdirs")
    parser.add_argument("--output", type=str, default="data/datasets/combined",
                        help="Output directory for combined parquets")
    args = parser.parse_args()

    base = Path(args.datasets_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # 自动发现所有数据集子目录（排除 combined 本身）
    ds_dirs = [d for d in base.iterdir()
               if d.is_dir() and d.name != "combined" and d.name != "abdct" and d.name != "abdmr"]

    print(f"Found datasets: {[d.name for d in ds_dirs]}")

    for split in ["train", "val", "test"]:
        all_dfs = []
        for ds_dir in ds_dirs:
            p = ds_dir / f"{split}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                all_dfs.append(df)
                print(f"  {ds_dir.name}/{split}: {len(df)} samples")
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            out_path = output / f"{split}.parquet"
            combined.to_parquet(out_path, index=False)
            print(f"Combined {split}: {len(combined)} samples -> {out_path}\n")


if __name__ == "__main__":
    main()
