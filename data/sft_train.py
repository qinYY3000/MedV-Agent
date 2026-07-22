"""
SFT 训练脚本 — 基于 Llama-Factory
=================================
原论文 MedSAM-Agent 的 SFT 阶段使用 Llama-Factory (见 README 致谢),
但代码未开源。本脚本补全 SFT 训练流程。

前置条件:
  1. 安装 Llama-Factory: pip install llamafactory[torch,metrics]
  2. 已运行 generate_trajectories.py 生成轨迹 JSON
  3. 下载 Qwen3-VL-8B-Instruct 模型权重

流程:
  1. 将轨迹 JSON 转为 Llama-Factory 的 sharegpt 格式
  2. 编写 Llama-Factory 数据集描述 (dataset_info.json)
  3. 编写训练 YAML 配置
  4. 运行训练

用法:
python data/sft_train.py \
    --trajectories data/sft_data/medsam_agent_sft.json \
    --model-path /mnt/workspace/Qwen3-VL-8B-Instruct \
    --output-dir data/sft_data \
    --llamafactory-dir /mnt/workspace/LlamaFactory
"""

import json
import os
import argparse
import base64
import io
from pathlib import Path
from PIL import Image
from collections import defaultdict


# ============================================================
# 1. 轨迹 JSON → Llama-Factory sharegpt 格式
# ============================================================

def b64_to_image(b64_str: str) -> Image.Image:
    """base64 → PIL Image。"""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def convert_to_sharegpt(trajectories: list, output_dir: str):
    """将轨迹 JSON 转为 Llama-Factory 的 sharegpt 格式。
    
    Llama-Factory sharegpt 格式:
    {
        "conversations": [
            {"from": "human", "value": "<image>Classify this image."},
            {"from": "gpt", "value": "<tool_call>...</tool_call>"}
        ],
        "images": ["path/to/image1.png", "path/to/image2.png"]
    }
    
    注意: Llama-Factory 要求图片存为文件, 不能用 base64 内联。
    所以这里需要把 base64 图片解码保存为文件。
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    sharegpt_data = []
    img_counter = 0
    
    for traj in trajectories:
        sample_id = traj["sample_id"]
        task_type = traj["task_type"]
        messages = traj["messages"]
        
        conversations = []
        image_paths = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # 映射 role
            if role == "system":
                # Llama-Factory sharegpt 不单独处理 system, 拼到第一个 human
                continue
            elif role == "user":
                from_role = "human"
            elif role == "assistant":
                from_role = "gpt"
            else:
                continue
            
            # 处理 content (可能是 list 或 str)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "image":
                            # base64 图片 → 保存为文件
                            b64 = item.get("image", "")
                            if b64:
                                img = b64_to_image(b64)
                                img_path = images_dir / f"img_{img_counter:06d}.png"
                                img.save(img_path)
                                image_paths.append(str(img_path))
                                text_parts.append("<image>")
                                img_counter += 1
                        elif item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                value = "\n".join(text_parts)
            elif isinstance(content, str):
                value = content
            else:
                value = str(content)
            
            # system prompt 拼到第一个 human 消息前面
            if role == "user" and not conversations:
                system_msg = next(
                    (m for m in messages if m["role"] == "system"), None
                )
                if system_msg:
                    sys_content = system_msg["content"]
                    if isinstance(sys_content, str):
                        value = sys_content + "\n\n" + value
            
            conversations.append({"from": from_role, "value": value})
        
        entry = {
            "conversations": conversations,
            "images": image_paths if image_paths else None,
        }
        # 移除 None 的 images 字段 (Llama-Factory 不需要空列表)
        if not image_paths:
            entry.pop("images")
        
        sharegpt_data.append(entry)
    
    # 保存
    output_file = output_dir / "sft_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
    
    print(f"Converted {len(sharegpt_data)} trajectories to sharegpt format")
    print(f"  Images saved: {img_counter}")
    print(f"  Dataset: {output_file}")
    
    # 统计
    task_counts = defaultdict(int)
    for traj in trajectories:
        task_counts[traj["task_type"]] += 1
    print(f"  Task distribution: {dict(task_counts)}")
    
    return output_file


# ============================================================
# 2. Llama-Factory dataset_info.json
# ============================================================

DATASET_INFO = {
    "medsam_agent_sft": {
        "file_name": "sft_dataset.json",
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


def write_dataset_info(output_dir: str):
    """写 Llama-Factory 的 dataset_info.json。"""
    info_path = Path(output_dir) / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(DATASET_INFO, f, ensure_ascii=False, indent=2)
    print(f"Dataset info: {info_path}")


# ============================================================
# 3. Llama-Factory 训练 YAML 配置
# ============================================================

TRAIN_YAML_TEMPLATE = """### model
model_name_or_path: {model_path}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset: medsam_agent_sft
dataset_dir: {data_dir}
template: qwen3_vl_nothink
cutoff_len: 8192
max_samples: 100000
preprocessing_num_workers: 16
dataloader_num_workers: 4
overwrite_cache: true

### output
output_dir: saves/qwen3-vl-8b/full/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# val_size: 0.05
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
"""


def write_train_yaml(model_path: str, data_dir: str, output_dir: str):
    """写 Llama-Factory 训练配置。"""
    yaml_content = TRAIN_YAML_TEMPLATE.format(
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir
    )
    yaml_path = Path(output_dir) / "sft_train.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Train config: {yaml_path}")
    return str(yaml_path)


# ============================================================
# 4. LoRA 配置 (可选, 节省显存)
# ============================================================

LORA_YAML_TEMPLATE = """### model
model_name_or_path: {model_path}
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: medsam_agent_sft
dataset_dir: {data_dir}
template: qwen3_vl_nothink
cutoff_len: 8192
max_samples: 100000
preprocessing_num_workers: 16
dataloader_num_workers: 4
overwrite_cache: true

### output
output_dir: saves/qwen3-vl-8b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
"""


def write_lora_yaml(model_path: str, data_dir: str, output_dir: str):
    """写 LoRA 训练配置 (显存友好)。"""
    yaml_content = LORA_YAML_TEMPLATE.format(
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir
    )
    yaml_path = Path(output_dir) / "sft_train_lora.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"LoRA config: {yaml_path}")
    return str(yaml_path)


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data for Llama-Factory")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Path to trajectories JSON file (generate_trajectories.py output)")
    parser.add_argument("--sharegpt-json", type=str, default=None,
                        help="Path to sharegpt JSON (prepare_sharegpt.py output, already sharegpt format)")
    parser.add_argument("--output-dir", type=str, default="data/sft_data",
                        help="Output directory for SFT data (relative to project root)")
    parser.add_argument("--llamafactory-dir", type=str, default=None,
                        help="LlamaFactory root directory. If set, copies dataset + config there and adjusts paths.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Path to base model")
    parser.add_argument("--use-lora", action="store_true",
                        help="Generate LoRA config instead of full fine-tuning")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SFT Data Preparation for Llama-Factory")
    print("=" * 60)
    
    if not args.trajectories and not args.sharegpt_json:
        parser.error("Must provide either --trajectories or --sharegpt-json")
    
    # 如果是 sharegpt-json 模式（prepare_sharegpt.py 输出），跳过转换
    if args.sharegpt_json:
        print(f"\nMode: Direct sharegpt JSON (skip conversion)")
        print(f"  Input: {args.sharegpt_json}")
        with open(args.sharegpt_json, "r", encoding="utf-8") as f:
            trajectories = json.load(f)
        print(f"  Loaded {len(trajectories)} entries")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 复制为标准文件名
        import shutil
        target_json = output_dir / "sft_dataset.json"
        if not target_json.exists() or str(target_json.resolve()) != str(Path(args.sharegpt_json).resolve()):
            shutil.copy2(args.sharegpt_json, target_json)
        # 写 dataset_info 和 yaml
        write_dataset_info(str(output_dir))
        if args.llamafactory_dir:
            lf_dir = Path(args.llamafactory_dir)
            lf_data_dir = lf_dir / "data"
            lf_data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_json, lf_data_dir / "medsam_agent_sft.json")
            lf_info_path = lf_data_dir / "dataset_info.json"
            if lf_info_path.exists():
                with open(lf_info_path) as f:
                    lf_info = json.load(f)
            else:
                lf_info = {}
            lf_info.update(DATASET_INFO)
            with open(lf_info_path, "w") as f:
                json.dump(lf_info, f, indent=2)
            print(f"Copied to LlamaFactory: {lf_data_dir}")
            if args.use_lora:
                yaml_path = write_lora_yaml(args.model_path, str(lf_data_dir), str(output_dir))
            else:
                yaml_path = write_train_yaml(args.model_path, str(lf_data_dir), str(output_dir))
        else:
            if args.use_lora:
                yaml_path = write_lora_yaml(args.model_path, str(output_dir), str(output_dir))
            else:
                yaml_path = write_train_yaml(args.model_path, str(output_dir), str(output_dir))
        print(f"\nTraining YAML: {yaml_path}")
        return
    
    # 传统模式: 加载轨迹 → 转 sharegpt
    if not args.trajectories:
        parser.error("--trajectories required for legacy mode")
    
    # Step 1: 加载轨迹
    print(f"\nStep 1: Loading trajectories from {args.trajectories}")
    with open(args.trajectories, "r", encoding="utf-8") as f:
        trajectories = json.load(f)
    print(f"  Loaded {len(trajectories)} trajectories")
    
    # Step 2: 转为 sharegpt 格式
    print(f"\nStep 2: Converting to sharegpt format...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = convert_to_sharegpt(trajectories, str(output_dir))
    
    # Step 3: 写 dataset_info.json
    print(f"\nStep 3: Writing dataset_info.json...")
    write_dataset_info(str(output_dir))
    
    # Step 4: 如果指定了 LlamaFactory 目录，把数据复制过去
    if args.llamafactory_dir:
        lf_dir = Path(args.llamafactory_dir)
        lf_data_dir = lf_dir / "data"
        lf_data_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        # 复制数据集 JSON
        shutil.copy2(str(output_dir / "sft_dataset.json"), str(lf_data_dir / "medsam_agent_sft.json"))
        # 复制 dataset_info.json (追加到 LlamaFactory 已有的 dataset_info)
        lf_info_path = lf_data_dir / "dataset_info.json"
        if lf_info_path.exists():
            with open(lf_info_path) as f:
                lf_info = json.load(f)
        else:
            lf_info = {}
        lf_info.update(DATASET_INFO)
        with open(lf_info_path, "w") as f:
            json.dump(lf_info, f, indent=2)
        # 复制图片目录
        images_src = output_dir / "images"
        images_dst = lf_data_dir / "medsam_agent_images"
        if images_src.exists():
            if images_dst.exists():
                shutil.rmtree(str(images_dst))
            shutil.copytree(str(images_src), str(images_dst))
        print(f"Copied SFT data to {lf_data_dir}")
        
        # 更新 YAML 中的 dataset_dir
        if args.use_lora:
            yaml_path = write_lora_yaml(args.model_path, str(lf_data_dir), str(output_dir))
        else:
            yaml_path = write_train_yaml(args.model_path, str(lf_data_dir), str(output_dir))
        print(f"Training YAML updated with LlamaFactory data dir: {lf_data_dir}")
    else:
        # Step 4: 写训练配置
        print(f"\nStep 4: Writing training YAML...")
        if args.use_lora:
            yaml_path = write_lora_yaml(args.model_path, str(output_dir), str(output_dir))
        else:
            yaml_path = write_train_yaml(args.model_path, str(output_dir), str(output_dir))
    
    # Step 5: 打印训练命令
    print(f"\n{'='*60}")
    print("Training Command:")
    print(f"{'='*60}")
    
    if args.llamafactory_dir:
        cmd_hint = f"""
# 数据集已放入 LlamaFactory data 目录, 直接运行:
cd {args.llamafactory_dir}
llamafactory-cli train {yaml_path.absolute()}
"""
    else:
        cmd_hint = f"""
# 数据在项目目录下, 需要指定 dataset_dir 参数:
llamafactory-cli train {yaml_path}

# 或复制到 LlamaFactory data 目录后运行:
#   cp {output_dir}/sft_dataset.json <LLAMAFACTORY_DIR>/data/
#   cp -r {output_dir}/images <LLAMAFACTORY_DIR>/data/
#   cp {output_dir}/dataset_info.json <LLAMAFACTORY_DIR>/data/
#   cd <LLAMAFACTORY_DIR>
#   llamafactory-cli train {output_dir}/sft_train.yaml
"""
    print(cmd_hint)
    
    print(f"\nAfter SFT, use the checkpoint as REF_MODEL_PATH for RL training:")
    print(f"  REF_MODEL_PATH={output_dir}/sft_checkpoint")
    print(f"  bash RL-verl/recipe/medsam_agent/run_multi_task.sh")


if __name__ == "__main__":
    main()
