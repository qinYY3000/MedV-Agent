#!/bin/bash
set -x

# ============================================================
# 多任务 VLM Agent RL 训练脚本
# 支持: 分类/检测/分割/复合 四种任务
# 基于 GRPO 算法, Verl框架
#
# 显存适配: 192GB AMD MI300X (ROCm)
# 策略: vllm rollout + FSDP (无需 offload)
# ============================================================

# --- 模型配置 ---
REF_MODEL_PATH=${REF_MODEL_PATH:-"/mnt/workspace/LlamaFactory/saves/qwen3.5_sft_merged"}

# --- 数据集路径 ---
DATASET_TRAIN=${DATASET_TRAIN:-"data/datasets/subset/train.parquet"}
DATASET_VAL=${DATASET_VAL:-"data/datasets/subset/val.parquet"}

# --- 输出路径 ---
SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-"./output/verl_multi_task_checkpoints"}
PROJECT_NAME="multi_task_agent"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"multi_task_busi_test"}

# --- 训练超参 (192GB 显存, 宽裕配置) ---
ACTOR_LR=${ACTOR_LR:-1e-6}
SAVE_FREQ=${SAVE_FREQ:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TRAIN_BATCH=${TRAIN_BATCH:-4}
N_GPU=${N_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-4}             # 4条轨迹 (显存够)
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}

# --- 显存优化开关 (192GB 默认关闭, 不需要 offload) ---
PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-False}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}     # vllm 显存占比

# --- 路径计算 ---
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RL_VERL_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LOG_DIR="${RL_VERL_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# --- 工具配置 (多任务) ---
TOOL_CONFIG="/mnt/workspace/MedSAM-Agent/RL-verl/recipe/medsam_agent/configs/multi_task_tools_config.yaml"
RECIPE_DIR="/mnt/workspace/MedSAM-Agent/RL-verl/recipe/medsam_agent"
CPR_REWARD="${RECIPE_DIR}/cpr_reward.py"
MULTI_TASK_DATASET="${RECIPE_DIR}/multi_task_dataset.py"
MULTI_TASK_AGENT_LOOP="${RECIPE_DIR}/multi_task_agent_loop.py"

echo "============================================"
echo "Multi-Task Agent RL Training (192GB ROCm)"
echo "============================================"
echo "Model:      $REF_MODEL_PATH"
echo "Train data: $DATASET_TRAIN"
echo "Val data:   $DATASET_VAL"
echo "Save dir:   $SAVE_CHECKPOINT_DIR"
echo "GPU count:  $N_GPU"
echo "Rollout:    vllm"
echo "Rollout n:  $ROLLOUT_N"
echo "Batch size: $TRAIN_BATCH"
echo "Offload:    param=$PARAM_OFFLOAD optimizer=$OPTIMIZER_OFFLOAD"
echo "============================================"

# --- 启动训练 ---
# AMD ROCm 环境
export HIP_VISIBLE_DEVICES=0
export ROCM_PATH=/opt/rocm
export RAY_DISABLE_DASHBOARD=1
export RAY_raylet_start_wait_time_s=120
# 显存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 清理旧的 ray 进程
ray stop --force 2>/dev/null
rm -rf /tmp/ray/*
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name='medsam_agent' \
    data.train_files=${DATASET_TRAIN} \
    "data.val_files=[${DATASET_VAL}]" \
    data.train_batch_size=${TRAIN_BATCH} \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    +data.custom_cls.name=multi_task_dataset \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG} \
    custom_reward_function.path=${CPR_REWARD} \
    custom_dataset.path=${MULTI_TASK_DATASET} \
    agent_loop.path=${MULTI_TASK_AGENT_LOOP} \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=500 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
