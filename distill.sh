#!/bin/bash
# ================================================================
# Java 蒸馏 — Deepin/RTX 4060 版
# 用法: bash distill.sh
# 需要: Deepin Linux + RTX 4060 (8GB) + Ollama + 50GB 磁盘
# ================================================================
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🪶 千羽 Java 蒸馏 — Deepin + RTX 4060${NC}"
echo ""

# ===== 0. 系统环境检查 =====
echo "===== 环境检查 ====="

# 检查 GPU
if ! nvidia-smi &>/dev/null; then
    echo "❌ 没检测到 NVIDIA 驱动！"
    echo "   Deepin 安装驱动:"
    echo "   sudo apt install nvidia-driver nvidia-cuda-toolkit"
    echo "   然后重启: sudo reboot"
    exit 1
fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "✅ GPU: $GPU"

# 检查 CUDA
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "未知")
echo "✅ CUDA: $CUDA_VER"

# 检查 Python
PY_VER=$(python3 --version 2>/dev/null || echo "未安装")
echo "✅ $PY_VER"

# 检查磁盘
DISK=$(df -h . | tail -1 | awk '{print $4}')
echo "✅ 磁盘可用: $DISK"

# 安装系统依赖 (Deepin/Debian)
echo ""
echo "===== 安装系统依赖 ====="
if ! dpkg -l build-essential &>/dev/null 2>&1; then
    echo "安装 build-essential..."
    sudo apt update -qq && sudo apt install -y -qq build-essential python3-pip python3-venv git curl
fi

# 检查 pip
if ! python3 -m pip --version &>/dev/null 2>&1; then
    echo "安装 pip..."
    sudo apt install -y -qq python3-pip
fi

# 创建虚拟环境（避免污染系统 Python）
if [ ! -d "venv" ]; then
    echo "创建 Python 虚拟环境..."
    python3 -m venv venv
fi
source venv/bin/activate

# ===== 1. 安装 Python 依赖 =====
echo ""
echo "===== 1/6 安装 Python 包 ====="

# Hugging Face 镜像加速
export HF_ENDPOINT=https://hf-mirror.com
echo "✅ HF 镜像: $HF_ENDPOINT""

# 确定 CUDA 版本对应的 PyTorch index
CUDA_MAJOR=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1 2>/dev/null || echo "12")
echo "CUDA 主版本: $CUDA_MAJOR"

if [ "$CUDA_MAJOR" = "12" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [ "$CUDA_MAJOR" = "11" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi

pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" 2>&1 | tail -1
pip install transformers datasets accelerate peft bitsandbytes scipy -q 2>&1 | tail -1

# 验证 CUDA
python3 << 'PYCHECK'
import torch
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
PYCHECK

# ===== 2. 获取 LLaMA-Factory =====
echo ""
echo "===== 2/6 LLaMA-Factory ====="
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e ".[torch]" -q 2>&1 | tail -1
cd ..

# ===== 3. 语料 =====
echo ""
echo "===== 3/6 语料 ====="
if [ ! -f "java-corpus-sharegpt.json" ]; then
    curl -sL "https://raw.githubusercontent.com/youdianwuliao/java-distill-corpus/main/java-corpus-sharegpt.json" \
        -o java-corpus-sharegpt.json
fi
cp java-corpus-sharegpt.json LLaMA-Factory/data/

python3 << 'PY'
import json
with open("LLaMA-Factory/data/dataset_info.json") as f:
    info = json.load(f)
info["java_corpus"] = {
    "file_name": "java-corpus-sharegpt.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
with open("LLaMA-Factory/data/dataset_info.json", "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print("✅ 数据集已注册")
PY

# ===== 4. 训练 =====
echo ""
echo "===== 4/6 训练（约 25-35 分钟）====="
echo -e "${YELLOW}⚠️ 插电 + 垫高散热！${NC}"

export HF_ENDPOINT=https://hf-mirror.com

cd LLaMA-Factory

llamafactory-cli train \
    --model_name_or_path "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --dataset java_corpus \
    --template qwen \
    --finetuning_type lora \
    --quantization_bit 4 \
    --quantization_type nf4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,v_proj \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 5 \
    --save_steps 1000 \
    --max_grad_norm 1.0 \
    --output_dir ../output/java-expert \
    --fp16 \
    --gradient_checkpointing \
    --max_length 2048

cd ..

# ===== 5. 合并 =====
echo ""
echo "===== 5/6 合并模型 ====="
cd LLaMA-Factory
llamafactory-cli export \
    --model_name_or_path "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --adapter_name_or_path ../output/java-expert \
    --template qwen \
    --export_dir ../java-expert-merged \
    --export_size 2 \
    --export_legacy_format false
cd ..

# ===== 6. 导入 Ollama =====
echo ""
echo "===== 6/6 Ollama ====="

cat > Modelfile << 'OLLAMA'
FROM ./java-expert-merged

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """你是千羽蒸馏的 Java 专家助手。精通 Spring Boot、MyBatis、Redis、MySQL、并发编程、JVM 调优。回答简洁准确，代码示例带注释。"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
OLLAMA

ollama create java-expert -f Modelfile
deactivate

echo ""
echo "================================================"
echo -e "  ${GREEN}🎉 蒸馏完成！${NC}"
echo ""
echo "  测试:"
echo "    ollama run java-expert"
echo "    > Spring Boot 全局异常处理怎么写？"
echo ""
echo "  产出:"
echo "    完整模型: ./java-expert-merged/"
echo "    LoRA权重: ./output/java-expert/"
echo "================================================"
