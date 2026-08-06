#!/bin/bash
set -ex

# ============================================================
# AMD ROCm 环境一键 setup (192GB MI300X)
# 解决所有已知兼容性问题
# ============================================================

echo "=== 1. 检查 ROCm 环境 ==="
rocm-smi --showproductname 2>/dev/null || echo "rocm-smi not found"
cat /opt/rocm/.info/version 2>/dev/null || echo "rocm version unknown"
python3 -c "import torch; print('torch:', torch.__version__, 'hip:', torch.version.hip)" 2>/dev/null || echo "torch not installed yet"

echo "=== 2. 安装 ROCm 版 torch/torchvision/torchaudio ==="
python3 -m pip install torch==2.10.0+rocm7.0 torchvision==0.25.0+rocm7.0 torchaudio==2.10.0+rocm7.0 \
    --index-url https://download.pytorch.org/whl/rocm7.0

echo "=== 3. 验证 torch + ROCm ==="
python3 -c "import torch; print('torch:', torch.__version__, 'hip:', torch.version.hip, 'gpu:', torch.cuda.is_available())"

echo "=== 4. 安装/修复 openai 版本 ==="
python3 -m pip install "openai==1.51.0" -q

echo "=== 5. 卸载 flash-attn (ROCm 不需要, 用 sdpa) ==="
python3 -m pip uninstall -y flash-attn 2>/dev/null || true

echo "=== 6. 卸载 trl (GRPO 不需要) ==="
python3 -m pip uninstall -y trl 2>/dev/null || true

echo "=== 7. 安装 vllm (ROCm 版, 在 torch 之后装) ==="
# vllm 需要匹配 torch 版本, 装完 torch 再装
python3 -m pip install vllm -q 2>/dev/null || {
    echo "  vllm 自动安装失败, 尝试指定版本..."
    python3 -m pip install "vllm==0.8.4" -q 2>/dev/null || echo "  vllm 安装失败, 后面用 HF rollout 兜底"
}
python3 -c "from vllm import LLM; print('vllm: OK')" 2>/dev/null || echo "  vllm import 失败, 训练脚本可切 HF rollout"

echo "=== 8. 修复 verl 代码兼容性 ==="
cd /mnt/workspace/MedSAM-Agent/RL-verl

# 8a. AutoModelForVision2Seq → AutoModelForImageTextToText (transformers 4.46+)
echo "  patch: AutoModelForVision2Seq..."
if grep -q "AutoModelForVision2Seq" verl/workers/fsdp_workers.py 2>/dev/null; then
    # 只在 import 行后面加 alias, 不改用法
    python3 -c "
p = 'verl/workers/fsdp_workers.py'
s = open(p).read()
# 如果还没 patch 过
if 'AutoModelForVision2Seq = AutoModelForImageTextToText' not in s:
    # 从 import 中移除 AutoModelForVision2Seq
    s = s.replace('            AutoModelForVision2Seq,\n', '')
    # 在 import 块后加 alias
    s = s.replace(
        '        from verl.utils.model import',
        '        AutoModelForVision2Seq = AutoModelForImageTextToText\n        from verl.utils.model import',
        1
    )
    open(p, 'w').write(s)
    print('  fsdp_workers.py patched')
else:
    print('  fsdp_workers.py already patched')
"
fi

# 8b. 其他文件的全局替换
grep -rl "AutoModelForVision2Seq" --include="*.py" verl/ 2>/dev/null | while read f; do
    if [[ "$f" != *"fsdp_workers.py" ]]; then
        sed -i 's/AutoModelForVision2Seq/AutoModelForImageTextToText/g' "$f"
        echo "  patched: $f"
    fi
done

# 8c. attn_implementation override (让 override_config 生效)
echo "  patch: attn_implementation override..."
if grep -q 'attn_implementation="flash_attention_2"' verl/workers/fsdp_workers.py 2>/dev/null; then
    python3 -c "
p = 'verl/workers/fsdp_workers.py'
s = open(p).read()
s = s.replace(
    'attn_implementation=\"flash_attention_2\"',
    'attn_implementation=override_model_config.get(\"attn_implementation\", \"flash_attention_2\")',
    1
)
open(p, 'w').write(s)
print('  attn_implementation patched')
"
fi

# 8d. trl import 加 try/except (如果 trl 被重装了)
echo "  patch: trl try/except..."
if grep -q "from trl import AutoModelForCausalLMWithValueHead" verl/models/transformers/monkey_patch.py 2>/dev/null; then
    python3 -c "
p = 'verl/models/transformers/monkey_patch.py'
s = open(p).read()
old = '''    if is_trl_available():
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore

        def state_dict(self, *args, **kwargs):
            return torch.nn.Module.state_dict(self, *args, **kwargs)

        AutoModelForCausalLMWithValueHead.state_dict = state_dict
        print(\"Monkey patch state_dict in AutoModelForCausalLMWithValueHead. \")'''
new = '''    if is_trl_available():
        try:
            from trl import AutoModelForCausalLMWithValueHead  # type: ignore

            def state_dict(self, *args, **kwargs):
                return torch.nn.Module.state_dict(self, *args, **kwargs)

            AutoModelForCausalLMWithValueHead.state_dict = state_dict
            print(\"Monkey patch state_dict in AutoModelForCausalLMWithValueHead. \")
        except ImportError:
            pass'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('  trl try/except patched')
else:
    print('  trl already patched or pattern not found')
"
fi

# 8e. vllm lora 导入路径 (vllm 0.11+ 改了路径)
echo "  patch: vllm lora import..."
if grep -q "from vllm.lora.models import LoRAModel" verl/utils/vllm/utils.py 2>/dev/null; then
    python3 -c "
p = 'verl/utils/vllm/utils.py'
s = open(p).read()
old = 'from vllm.lora.models import LoRAModel'
new = '''try:
    from vllm.lora.models import LoRAModel
except ImportError:
    try:
        from vllm.lora.lora_model import LoRAModel
    except ImportError:
        from vllm.lora.lora import LoRAModel as LoRAModel'''
if old in s:
    s = s.replace(old, new, 1)
    open(p, 'w').write(s)
    print('  vllm lora import patched')
else:
    print('  vllm lora already patched')
"
fi

echo "=== 9. 安装 verl ==="
python3 -m pip install -e . --no-build-isolation -q 2>/dev/null || echo "verl install skipped (may already be installed)"

echo "=== 10. 验证 ==="
python3 -c "import torch; print('torch:', torch.__version__, 'hip:', torch.version.hip)"
python3 -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>/dev/null || echo "torchvision: not installed"
python3 -c "from vllm import LLM; print('vllm: OK')" 2>/dev/null || echo "vllm: import failed (may need ROCm build)"
python3 -c "from transformers import AutoModelForImageTextToText; print('transformers: OK')"
python3 -c "import verl; print('verl: OK')" 2>/dev/null || echo "verl: import issue"
python3 -c "import py_compile; py_compile.compile('verl/workers/fsdp_workers.py', doraise=True); print('fsdp_workers: syntax OK')"

echo ""
echo "=== Setup 完成 ==="
echo "如果 vllm import 失败, 需要安装 ROCm 版 vllm:"
echo "  pip install vllm (会自动检测 ROCm)"
echo "  或指定版本: pip install vllm==0.8.4"
echo ""
echo "然后运行训练:"
echo "  bash RL-verl/recipe/medsam_agent/run_multi_task.sh"
