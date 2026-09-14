cd /mnt/workspace/MedSAM-Agent

# 可选：清理旧 combined，避免误用历史混合集
rm -rf data/datasets/combined

# RL parquet：只生成三类当前使用的数据
python data/prepare_all_datasets.py \
  --busi data/Dataset_BUSI_with_GT \
  --kvasir data/kvasir-seg \
  --tn3k data/tn3k \
  --output data/datasets

# SFT 数据：只生成三类当前使用的数据
python data/prepare_sharegpt.py \
  --source busi data/Dataset_BUSI_with_GT \
  --source kvasir data/kvasir-seg \
  --source tn3k data/tn3k \
  --output data/sft_data \
  --llamafactory-dir /mnt/workspace/LlamaFactory

# 白名单合并，避免混入 CT/MR/X-ray
python data/combine_parquet.py \
  --datasets-dir data/datasets \
  --include busi kvasir tn3k \
  --output data/datasets/combined
