# MedV-Agent 多任务医学视觉 Agent 扩展

![MedV-Agent](./assets/logo.png)

## 项目定位

本项目基于上游 `MedSAM-Agent` 的交互式分割能力，扩展为面向多工具医学视觉分析的任务条件化 Agent：

```text
全图筛查分类 → 候选病灶检测 → 交互式精细分割 → ROI 区域级表征 → 自主停止
```

- VLM 根据用户任务、当前视觉状态与工具反馈，动态选择 `classify`、`detect`、`add_bbox`、`add_point` 或 `stop_action`；
- 对明确正常的筛查样本，允许安全提前结束，避免无意义的定位和分割；
- 分类、检测与分割后端均以独立服务封装，可替换、可独立部署；
- 使用规则驱动的 **CPR v2（Clinical Process Reward v2）** 为 GRPO 提供任务质量、过程增益、工具协同、交互成本与安全风险的联合奖励。

本项目的推荐研究主题为：

> **面向多工具医学视觉智能体的任务条件化临床过程奖励方法**  
> **Task-Conditioned Clinical Process Reward for Multi-Tool Medical Vision Agents**

## 核心方法

### 多工具 Agent

| 工具 | 后端 | 作用 |
| --- | --- | --- |
| `classify` | BioMedCLIP | 全图筛查或 ROI 区域级分类 |
| `detect` | Grounding DINO | 文本驱动的候选目标定位 |
| `add_bbox` / `add_point` | IMISNet / MedSAM2 | 交互式分割初始化与迭代细化 |
| `stop_action` | Agent 策略 | 根据当前任务质量自主停止 |

复合任务并非固定流水线。Agent 可以根据任务类型及中间反馈决定是否继续调用工具、是否复用检测框或分割区域，以及何时停止。

### CPR v2：任务条件化临床过程奖励

CPR v2 不只比较最终预测与真值，还将 Agent 轨迹表示为工具事件序列，联合评估：

- **终局质量**：分类、检测、分割的最终任务质量；
- **事件级状态增益**：每次工具调用是否确实改善当前状态，例如分割前后 IoU/Dice 的变化；
- **跨工具信息流协同**：筛查分类→检测、检测框→分割提示、分割 mask→ROI 分类是否被有效复用；
- **任务条件化策略与格式**：不同任务只评价其应执行的有效动作，并检查工具调用格式；
- **交互成本与停止质量**：惩罚重复、无关、失败和无效细化调用，鼓励在质量足够时及时停止；
- **规则式临床安全约束**：对恶性/异常→正常、目标漏检、低质量分割及正常样本假阳性施加非对称惩罚。

当前实现为低开销的规则版 CPR v2，奖励裁剪到 `[-1, 1]`。学习式 Clinical Judge、推理置信度评分与医生一致性评价属于后续工作，尚未作为当前结果声明。详细设计见 [`docs/PROJECT_INNOVATIONS.md`](docs/PROJECT_INNOVATIONS.md)。

## 数据与训练链路

项目覆盖六个数据集、五种模态：

| 数据集 | 模态 | 核心任务 |
| --- | --- | --- |
| BUSI | 超声 | 分类、检测、分割 |
| Kvasir-SEG | 内镜 | 分类、检测、分割 |
| COVID-19 | X-ray | 分类、分割 |
| TN3K | 超声 | 分类、检测、分割 |
| AbdomenCT | CT | 多器官分类、检测、分割 |
| AbdomenMR | MR | 多器官分类、检测、分割 |

```text
SFT：原图目录 → ShareGPT 多轮工具轨迹 → LlamaFactory LoRA SFT
RL ：统一样本字段 → parquet → CT/MR 3D NIfTI 转 2D 切片 → combined parquet → verl GRPO
```

CT/MR 数据会将每个 3D NIfTI 体数据转换为代表性 2D 切片，并为同一切片内的每个器官生成独立的 mask、bbox 与训练样本。数据处理的详细步骤见 [`PLAN_MULTIAGENT.md`](PLAN_MULTIAGENT.md)。

## 快速开始

### 1. 准备依赖与模型

本项目包含 NVIDIA CUDA 与 AMD ROCm 的训练/服务适配脚本；请根据实际硬件安装匹配的 PyTorch、vLLM/SGLang 与后端模型依赖。至少准备：

- Grounding DINO 权重及 `bert-base-uncased` tokenizer；
- BioMedCLIP 权重及其文本编码器缓存；
- IMISNet 或 MedSAM2 权重；
- 用于 SFT/RL 的 Qwen-VL 系列模型及 LlamaFactory、verl 环境。

```bash
pip install -r requirements.txt
```

> 具体版本组合与离线模型准备请以 [`PLAN_MULTIAGENT.md`](PLAN_MULTIAGENT.md) 和对应启动脚本为准。ROCm 环境需使用匹配 ROCm 版本的 PyTorch wheel，并建议使用 SDPA，避免将 CUDA 版 `flash-attn` 安装到 ROCm 环境。

### 2. 生成 SFT 数据

```bash
python data/prepare_sharegpt.py \
  --source busi data/Dataset_BUSI_with_GT \
  --source kvasir data/kvasir-seg \
  --source covid data/covid-19 \
  --source tn3k data/tn3k \
  --source abdct data/datasets/abdct_2d \
  --source abdmr data/datasets/abdmr_2d \
  --output data/sft_data \
  --llamafactory-dir /mnt/workspace/LlamaFactory
```

SFT 轨迹包括：`classify → stop`、`detect → stop`、`add_bbox → add_point → stop`，以及筛查/检测/分割/ROI 分类组成的复合轨迹。对 CT/MR 器官任务，目标名称会显式写入 prompt，例如 “Segment the liver in this CT/MR image.”。

### 3. 生成并合并 RL parquet

先生成四个原生 2D 数据集 parquet，再将 CT/MR 的 3D 数据转换为 2D 切片 parquet：

```bash
python data/prepare_all_datasets.py \
  --busi data/Dataset_BUSI_with_GT \
  --kvasir data/kvasir-seg \
  --covid data/covid-19 \
  --tn3k data/tn3k \
  --output data/datasets

python data/extract_3d_slices.py \
  --input-dir data/datasets/abdct \
  --output-dir data/datasets/abdct_2d \
  --slices-per-volume 3

python data/extract_3d_slices.py \
  --input-dir data/datasets/abdmr \
  --output-dir data/datasets/abdmr_2d \
  --slices-per-volume 3

python data/combine_parquet.py \
  --datasets-dir data/datasets \
  --output data/datasets/combined
```

`combined/train.parquet`、`combined/val.parquet` 与 `combined/test.parquet` 是多数据集 RL 采样入口。当前 parquet 预处理链路保存图像与 mask 路径；对应原始图像/切片和 mask 文件必须在训练节点按记录路径可访问，除非另行完成图片 bytes 内嵌转换。

### 4. 启动三个工具服务

分别在独立终端启动：

```bash
bash RL-verl/api_server/run_api.sh                 # segmentation :8265
bash RL-verl/api_server/run_detection_api.sh       # detection    :8266
bash RL-verl/api_server/run_classification_api.sh  # classification:8267
```

健康检查：

```bash
curl http://localhost:8265/health
curl http://localhost:8266/health
curl http://localhost:8267/health
```

### 5. SFT 与 GRPO

SFT 使用 LlamaFactory 对 ShareGPT 工具轨迹进行 LoRA 微调；合并 LoRA 权重后，使用 verl 进行多轮 GRPO：

```bash
# 192GB ROCm 配置示例
bash RL-verl/recipe/medsam_agent/run_multi_task.sh

# 单卡 24GB NVIDIA 配置示例（4B 模型）
bash RL-verl/recipe/medsam_agent/run_nvidia_24g.sh
```

核心训练配置包括：

- `algorithm.adv_estimator=grpo`；
- `actor_rollout_ref.rollout.multi_turn.enable=True`；
- 自定义 Agent loop：`multi_task_agent_loop.py`；
- 自定义奖励：`cpr_reward.py`；
- 多任务数据集：`multi_task_dataset.py`。

## 当前实现状态

已完成：

- 分类、检测、交互式分割与复合任务的统一 Agent loop；
- Grounding DINO、BioMedCLIP、IMISNet/MedSAM2 三类后端 API；
- 六数据集、五模态以及 CT/MR `3D → 2D` 数据处理脚本；
- 从分割标注自动派生 bbox、类别与多任务 SFT 轨迹；
- CPR v2 的终局质量、事件增益、协同、策略/格式、动作成本与规则式安全项；
- CPR v2 与奖励信息透传的局部测试。

尚待正式实验验证：

- 含 Ray、模型权重与三类工具服务的完整 GRPO 端到端稳定训练；
- CPR v2 相对终局奖励的稳定性能增益、消融、跨数据集泛化与安全性结论；
- 学习式 Clinical Judge 及医生一致性评价。

因此，请勿将当前仓库表述为已完成大规模 GRPO 收敛或已获得稳定性能提升的最终系统。

## 文档导航

- [`docs/PROJECT_INNOVATIONS.md`](docs/PROJECT_INNOVATIONS.md)：项目贡献边界、CPR v2 公式、实现状态与实验建议；
- [`PLAN_MULTIAGENT.md`](PLAN_MULTIAGENT.md)：多任务数据、后端服务、SFT/GRPO 实施流程；
- [`docs/CPR_REWARD_DESIGN.md`](docs/CPR_REWARD_DESIGN.md)：早期 CPR 概念设计与后续扩展方向；
- [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md)：基于当前实现状态的面试问答口径；
- [`docs/RESUME_PROJECT.md`](docs/RESUME_PROJECT.md)：简历项目描述模板。

## Acknowledgements

本项目建立在以下开源工作之上：

- [MedSAM-Agent](https://arxiv.org/abs/2602.03320)
- [verl](https://github.com/verl-project/verl)
- [LlamaFactory](https://github.com/hiyouga/LlamaFactory)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [MedSAM2](https://medsam2.github.io/)
- [IMISNet](https://github.com/uni-medical/IMIS-Bench)
