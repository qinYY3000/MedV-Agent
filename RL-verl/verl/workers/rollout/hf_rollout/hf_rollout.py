"""
HuggingFace Native Rollout for AMD ROCm
=========================================
不依赖 vllm/sglang，直接用 transformers 推理。
适合 AMD GPU 环境（vllm/sglang CUDA 依赖不兼容时使用）。

性能: 比 vllm/sglang 慢 3-5 倍，但完全兼容。
"""

import os
import torch
import asyncio
from typing import Generator, Optional
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

__all__ = ["HFRollout", "HFAsyncRollout"]


class HFRollout(BaseRollout):
    """HuggingFace 原生推理 rollout (sync 模式)。"""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_path = model_config.path
        self.dtype = getattr(torch, config.get("model_dtype", "bfloat16"))
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.6)
        self.max_num_batched_tokens = config.get("max_num_batched_tokens", 16384)
        self.enforce_eager = config.get("enforce_eager", True)
        self.n = config.get("n", 1)
        self.sampling_params = {
            "max_new_tokens": config.get("max_response_length", 8192),
            "do_sample": True,
            "temperature": config.get("temperature", 1.0),
            "top_p": config.get("top_p", 0.9),
        }

        print(f"[HFRollout] Loading model from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=config.get("attn_implementation", "sdpa"),
        )
        if self.enforce_eager:
            self.model.eval()
        print(f"[HFRollout] Model loaded on {next(self.model.parameters()).device}")

        # 用于 update_weights 的占位
        self._weight_cache = {}

    async def resume(self, tags: list[str]):
        """恢复权重（HF 模式不需要 KV cache 管理）。"""
        pass

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """从 FSDP actor 同步权重到 rollout 模型。"""
        updated = 0
        for name, tensor in weights:
            # 直接加载到模型
            try:
                param = dict(self.model.named_parameters()).get(name)
                if param is not None:
                    param.data.copy_(tensor.to(param.device, param.dtype))
                    updated += 1
            except Exception as e:
                print(f"[HFRollout] Warning: failed to update {name}: {e}")
        print(f"[HFRollout] Updated {updated} parameters")

    async def release(self):
        """释放显存。"""
        if hasattr(self, "model"):
            del self.model
            torch.cuda.empty_cache()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """批量生成序列。"""
        from verl.utils.torch_functional import pad_sequence

        batch = prompts.batch
        input_ids_list = batch["input_ids"]  # List[Tensor] 或 (B, L)
        attention_mask_list = batch.get("attention_mask")
        images = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")

        # 转 list
        if isinstance(input_ids_list, torch.Tensor):
            input_ids_list = [ids for ids in input_ids_list]
        if attention_mask_list is not None and isinstance(attention_mask_list, torch.Tensor):
            attention_mask_list = [m for m in attention_mask_list]

        device = next(self.model.parameters()).device
        all_responses = []

        for i in range(len(input_ids_list)):
            input_ids = input_ids_list[i].to(device)
            attn_mask = attention_mask_list[i].to(device) if attention_mask_list else None

            # 生成 n 条轨迹
            responses = []
            for _ in range(self.n):
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=input_ids.unsqueeze(0),
                        attention_mask=attn_mask.unsqueeze(0) if attn_mask is not None else None,
                        **self.sampling_params,
                    )
                # 只取新生成的部分
                new_tokens = output[0][input_ids.shape[0]:]
                responses.append(new_tokens)

            all_responses.append(responses)

        # 打包返回
        return self._pack_output(all_responses, input_ids_list)

    def _pack_output(self, all_responses, input_ids_list):
        """把生成结果打包成 DataProto。"""
        # 简化版: 展平后 pad
        flat_responses = []
        for responses in all_responses:
            flat_responses.extend(responses)

        max_len = max(r.shape[0] for r in flat_responses)
        padded = torch.full(
            (len(flat_responses), max_len),
            self.processor.tokenizer.pad_token_id or 0,
            dtype=flat_responses[0].dtype,
            device=flat_responses[0].device,
        )
        for i, r in enumerate(flat_responses):
            padded[i, :r.shape[0]] = r

        output = DataProto()
        output.batch = {"responses": padded}
        return output


class HFAsyncRollout(HFRollout):
    """HuggingFace 异步 rollout (实际是 sync 包装成 async)。"""

    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        return self.generate_sequences(prompts)
