# VLM 多任务视觉 Agent 实施方案

> 基于 MedSAM-Agent 框架，以高精度病灶分割为目标，利用筛查分类、检测定位和 ROI 表征形成临床闭环
> 支持 6 个数据集、5 种模态（超声/内镜/X-ray/CT/MR）、5 个解剖部位

---

## 一、项目目标

### 核心假设
> VLM 作为诊断辅助分割智能体，能以全图筛查分类提供语义先验、以检测提供空间先验，并通过交互式分割完成精细病灶量化；在复合病例中学习优于固定工具串联的策略。

### 设计原则
- **后端模型零训练**: 检测用 Grounding DINO、分类用 BioMedCLIP、分割用 IMISNet/MedSAM2
- **数据零额外标注**: 从分割 mask 自动派生检测 bbox 和分类标签
- **SFT 轨迹零人工**: 用规则模板 + GT 信息自动生成

---

## 二、数据集

### 2.1 数据集概览

| 数据集 | 模态 | 部位 | 标签 | 样本数 | 任务 |
|--------|------|------|------|--------|------|
| BUSI | 超声 | 乳腺 | benign/malignant/normal | 780 | 分类+检测+分割 |
| Kvasir-SEG | 内镜 | 结肠 | polyp | 1000 | 分类+检测+分割 |
| COVID-19 | X-ray | 胸部 | covid/lung_opacity/normal/pneumonia | 21165 | 分类+分割 |
| TN3K | 超声 | 甲状腺 | thyroid_nodule | 3493 | 分类+检测+分割 |
| AbdomenCT | CT | 腹部 | 13个器官(liver/kidney/spleen...) | 100vol→~500切片 | 分类+检测+分割 |
| AbdomenMR | MR | 腹部 | 13个器官 | 110vol→~550切片 | 分类+检测+分割 |

### 2.2 数据处理流程

```
Step 1: SFT 数据 (直接从图片 → sharegpt JSON)
  prepare_sharegpt.py → medsam_agent_sft.json → LlamaFactory 训练

Step 2: RL 数据 (parquet 格式, Verl 要求)
  prepare_all_datasets.py → parquet (3D NIfTI 路径)
  extract_3d_slices.py → 2D 切片 PNG + parquet (仅 CT/MR 需要)
  合并 → combined/train.parquet → Verl GRPO 训练
```

### 2.3 数据划分

```
训练集: 70%
验证集: 15%
测试集: 15%
统一 seed=42 切分
```

---

## 三、工具架构

### 3.1 整体架构

```
Agent（Qwen-VL 系列；当前快速验证配置为 Qwen3.5-4B）
    │
    ├── triage classify → BioMedCLIP API (:8267) [全图筛查]
    ├── detect          → Grounding DINO API (:8266) [候选定位]
    ├── segment         → IMISNet API (:8265) [边界精描]
    │     ├── add_bbox
    │     ├── add_point
    ├── ROI classify    → BioMedCLIP API (:8267) [病灶表征]
    └── stop_action     → 正常筛查可提前结束
```

### 3.2 后端模型

| 组件 | 模型 | 类型 | 是否需要训练 |
|------|------|------|:---:|
| 检测 | Grounding DINO (Swin-T) | 开放词汇检测 | ❌ |
| 分类 | BioMedCLIP (ViT-B/16) | 医学图文对比 | ❌ |
| 分割 | IMISNet (SAM-ViT-B + CLIP) | 交互式医学分割 | ❌ |

---

## 四、实施流程

### Phase 0: 下载预训练权重

```bash
# 1. Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git RL-verl/api_server/GroundingDINO
wget -P RL-verl/api_server/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
pip install -e RL-verl/api_server/GroundingDINO

# 2. BioMedCLIP
modelscope download --model microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --local_dir RL-verl/api_server/biomedclip_model
pip install open-clip-torch==2.27.0

# 3. IMISNet
modelscope download --model 1Junlong/IMIS-Net IMISNet-B.pth --local_dir models/

# 4. CLIP tokenizer (IMISNet 依赖)
modelscope download --model openai-mirror/clip-vit-base-patch32 --local_dir models/clip-vit-base-patch32

# 5. BERT tokenizer (Grounding DINO 依赖)
HF_ENDPOINT=https://hf-mirror.com python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"

modelscope download --model Qwen/Qwen3.5-4B --local_dir /mnt/workspace/Qwen3.5-4B
```

### Phase 1: 数据预处理

#### Step 1.1: SFT 数据生成

```bash
# CT/MR 先切片（仅一次）
pip install nibabel
python data/extract_3d_slices.py --input-dir data/datasets/abdct --output-dir data/datasets/abdct_2d --slices-per-volume 5
python data/extract_3d_slices.py --input-dir data/datasets/abdmr --output-dir data/datasets/abdmr_2d --slices-per-volume 5

# 当前超声扩展范围：BUSI、Group Breast、TN3K、SZU-BCH-TUS983。
# Group Breast 从有效实例标注帧中按视频序列均衡选择 1000 张。
cd /mnt/workspace/MedSAM-Agent
python data/prepare_sharegpt.py \
    --source busi data/Dataset_BUSI_with_GT \
    --source group_breast data/group_breast \
    --source tn3k data/tn3k \
    --source szu_bch_tus data/SZU-BCH-TUS983 \
    --group-breast-max-frames 1000 \
    --max-per-task 0 \
    --per-modality 0 \
    --output data/sft_data \
    --llamafactory-dir /mnt/workspace/LlamaFactory
```

#### Step 1.2: RL 数据生成 (parquet)

```bash
# 当前超声扩展范围：BUSI、Group Breast、TN3K、SZU-BCH-TUS983。
# Group Breast 从有效实例标注帧中按视频序列均衡选择 1000 张。
# 1. 扫描 4 个超声数据集 → parquet
python data/prepare_all_datasets.py \
    --busi data/Dataset_BUSI_with_GT \
    --group-breast data/group_breast \
    --group-breast-max-frames 1000 \
    --tn3k data/tn3k \
    --szu-bch-tus data/SZU-BCH-TUS983 \
    --output data/datasets

# 2. 仅合并本轮选定数据集；不要自动混入历史内镜、X-ray、CT、MR parquet。
python data/combine_parquet.py \
    --datasets-dir data/datasets \
    --include busi group_breast tn3k szu_bch_tus \
    --output data/datasets/combined
```

#### Step 1.3: 启动三个 API

```bash
# 终端1: 分割 API (IMISNet)
export HF_ENDPOINT=https://hf-mirror.com
pip install hydra-core==1.3.2 timm         pip install monai
bash RL-verl/api_server/run_api.sh              # :8265

# 终端2: 检测 API (Grounding DINO)
pip install yapf scikit-image supervision pycocotools addict
bash RL-verl/api_server/run_detection_api.sh    # :8266

# 终端3: 分类 API (BioMedCLIP)
pip install open-clip-torch==2.27.0
bash RL-verl/api_server/run_classification_api.sh  # :8267

# 验证
curl http://localhost:8265/health
curl http://localhost:8266/health
curl http://localhost:8267/health

```

### Phase 2: SFT 训练

#### Step 2.1: LlamaFactory 安装

```bash
pip install llamafactory[torch,metrics]
# 或从源码安装
git clone https://github.com/hiyouga/LlamaFactory.git /mnt/workspace/LlamaFactory
cd /mnt/workspace/LlamaFactory && pip install -e ".[torch,metrics]"
```

#### Step 2.2: 启动 SFT 训练 (LoRA)

```bash
# 1. 确保 PyTorch ROCm 版本正确
pip install torch==2.10.0+rocm7.0 torchvision==0.25.0+rocm7.0 torchaudio==2.10.0+rocm7.0 --index-url https://download.pytorch.org/whl/rocm7.0

# 2. 首次训练 (建议先用 max_samples: 2000 快速验证)
cd /mnt/workspace/LlamaFactory
export MIOPEN_ENABLED=0
export MIOPEN_DEBUG_CONV_FFT=0
export MIOPEN_DEBUG_CONV_DIRECT=0
export MIOPEN_DEBUG_CONV_WINOGRAD=0
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
llamafactory-cli train /mnt/workspace/MedSAM-Agent/data/sft_data/sft_train_4b.yaml

```

**训练策略 (针对 8 小时实例限制)**:

| 策略 | max_samples | epochs | 预计时间 | 用途 |
|------|:---:|:---:|------|------|
| 快速验证 | 2000 | 1 | ~30min | 确认流程跑通 |
| 小规模 | 20000 | 1 | ~2h | 验证模型学习 |
| 全量 | 100000 | 3 | ~6-8h | 正式训练 |

**断点续训**:
- checkpoint 每 500 步自动保存到对应模型的 LoRA 输出目录，例如 `saves/qwen3.5-4b/lora/sft/checkpoint-XXX/`
- 分词缓存 (.arrow) 在 `~/.cache/huggingface/datasets/` 下，实例重启后保留
- 下次训练时用 `--resume_from_checkpoint` 从上次 checkpoint 继续，跳过分词阶段

#### Step 2.3: 合并 LoRA 权重

```bash
llamafactory-cli export \
    --model_name_or_path /mnt/workspace/Qwen3.5-4B \
    --adapter_name_or_path /mnt/workspace/LlamaFactory/saves/qwen3.5-4b/lora/sft \
    --template qwen3_vl_nothink \
    --finetuning_type lora \
    --export_dir /mnt/workspace/LlamaFactory/saves/qwen3.5_sft_merged
```

#### Step 2.4: SFT 评估

```bash
cd /mnt/workspace/MedSAM-Agent && python data/eval_sft.py \
    --model-path /mnt/workspace/LlamaFactory/saves/qwen3.5_sft_merged \
    --source tn3k data/tn3k \
    --source abdct data/datasets/abdct_2d \
    --source abdmr data/datasets/abdmr_2d \
    --output data/sft_data/eval_results_covid.json \
    --num-samples 5
```

**验收标准**: tool_call 格式正确率 > 80%

**上传模型到 ModelScope**:
```bash
# 将 SFT 合并后的模型上传到 ModelScope
# 通过环境变量提供访问令牌，避免将敏感信息写入文档或仓库
modelscope upload ScholarChen20/Qwen3.5-4B \
    /mnt/workspace/LlamaFactory/saves/qwen3.5_sft_merged \
    --token "$MODELSCOPE_TOKEN"
```

### Phase 3: RL 训练 (GRPO)

#### Step 3.1: 确认三个 API 已运行

```bash
curl http://localhost:8265/health  # 分割
curl http://localhost:8266/health  # 检测
curl http://localhost:8267/health  # 分类
```

#### Step 3.2: 安装 Verl + AMD ROCm 兼容修复

```bash
cd RL-verl
pip install -e ".[sglang]" --no-build-isolation
```

> **注**: 以上修复针对镜像环境 `Ubuntu 22.04 + ROCm 7.2.1 + vLLM 0.20.1+rocm721`。

#### Step 3.3: 启动 RL 训练

```bash
# ROCm 环境建议使用 SDPA；不要安装 CUDA 版 flash-attn
cd /mnt/workspace/MedSAM-Agent
bash RL-verl/recipe/medsam_agent/run_multi_task.sh
```

**GRPO 核心配置**:
- `algorithm.adv_estimator=grpo`：使用 GRPO 优化多轮工具策略；
- `rollout.n=2`：每个 prompt 采样 2 条轨迹（子集验证配置，可按显存调整）；
- `rollout.name=vllm`：使用与 ROCm/PyTorch 匹配的 vLLM 版本；
- `multi_turn.enable=True`：开启多轮工具调用；
- `custom_reward_function.path=recipe/medsam_agent/cpr_reward.py`：启用规则版 CPR v2。

CPR v2 在终局质量之外，计算事件级状态增益、筛查→检测/检测→分割/分割→ROI 分类的信息流协同、任务条件化策略与格式、动作成本和规则式安全惩罚。当前已完成局部测试；完整 GRPO 收敛、终局性能增益与消融结论仍待正式实验验证。

### Phase 4: 评估

#### 评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| 筛查分类 | Sensitivity, Specificity, F1 | 是否提供可靠的病灶存在性/风险先验 |
| 检测 | IoU, mAP@0.5, Recall | 候选框对分割初始化的空间先验质量 |
| 分割 | Dice, IoU, HD95 | 核心终点：mask 精度与边界质量 |
| ROI 表征 | Accuracy, F1 | 基于分割区域的病灶性质判断 |
| 复合 | 端到端分割增益、漏诊率、平均步数 | 筛查—定位—分割—表征闭环 |
| 效率 | 平均步数 | 越少越好 |

#### 对比实验

| 实验 | 方法 | 目的 |
|------|------|------|
| A | Grounding DINO (零样本) | 检测基线 |
| B | BioMedCLIP (零样本) | 分类基线 |
| C | IMISNet (预训练) | 分割基线 |
| D | Agent (SFT) | 验证单任务 |
| E | Agent (SFT+RL) | 验证 RL 提升 |
| F | 固定临床串联（筛查→检测→分割→ROI 表征） | 诊断辅助分割基线 |
| G | Agent 临床闭环 | 验证动态工具调度与分割增益 |

---

## 五、项目结构

```
MedSAM-Agent/
├── data/
│   ├── prepare_sharegpt.py          # SFT 数据生成（6个数据集）
│   ├── prepare_all_datasets.py      # RL parquet 数据生成
│   ├── extract_3d_slices.py         # CT/MR 3D→2D 切片
│   ├── eval_sft.py                  # SFT 评估
│   ├── sft_train.py                 # SFT YAML 配置生成
│   ├── sft_data/                    # SFT 训练数据
│   └── datasets/                    # RL parquet 数据
├── RL-verl/
│   ├── api_server/
│   │   ├── segmentation_api.py      # IMISNet 分割 API (:8265)
│   │   ├── detection_api.py         # Grounding DINO 检测 API (:8266)
│   │   ├── classification_api.py    # BioMedCLIP 分类 API (:8267)
│   │   └── run_*.sh                 # API 启动脚本
│   ├── recipe/medsam_agent/
│   │   ├── run_multi_task.sh        # RL 训练启动脚本
│   │   ├── cpr_reward.py            # CPR 奖励函数
│   │   ├── multi_task_dataset.py    # RL 数据集
│   │   ├── multi_task_agent_loop.py # Agent 循环
│   │   └── configs/                 # Hydra 配置
│   └── verl/                        # Verl 框架
├── third_party/
│   ├── segment_anything/            # IMISNet/SAM
│   └── sam2/                        # MedSAM2 (可选)
└── PLAN_MULTIAGENT.md               # 本文件
```

---

## 六、时间线

```
Week 1: Phase 0 (权重下载) + Phase 1 (数据预处理 + API)
Week 2: Phase 2 (SFT 训练 + 评估)
Week 3-4: Phase 3 (RL 训练)
Week 5: Phase 4 (评估与实验)
```
