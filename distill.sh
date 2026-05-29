#!/bin/bash
# ================================================================
# Java 蒸馏模型 — 一键训练脚本
# 
# 用法:
#   bash distill.sh
#
# 需要:
#   - NVIDIA GPU (≥8GB 显存)
#   - 50GB 磁盘空间（模型下载 + 训练产出）
# ================================================================
set -e

echo "🪶 千羽 Java 蒸馏 — 开始"

# ===== 1. 环境 =====
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers datasets accelerate peft bitsandbytes -q

# 如果有 LLaMA-Factory 就用它，没有就用原生 transformers
if [ ! -d "LLaMA-Factory" ]; then
    echo "📦 安装 LLaMA-Factory..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" -q
    cd ..
fi

# ===== 2. 下载语料 =====
if [ ! -f "java-corpus-sharegpt.json" ]; then
    echo "📥 下载语料..."
    curl -sL "https://raw.githubusercontent.com/youdianwuliao/java-distill-corpus/main/java-corpus-sharegpt.json" \
        -o java-corpus-sharegpt.json
fi

cp java-corpus-sharegpt.json LLaMA-Factory/data/

# ===== 3. 注册数据集 =====
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

# ===== 4. 下载基座模型 =====
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
echo "📥 下载基座模型: $MODEL"

# ===== 5. 训练 =====
echo "🔥 开始蒸馏训练..."
cd LLaMA-Factory

llamafactory-cli train \
    --model_name_or_path "$MODEL" \
    --dataset java_corpus \
    --template qwen \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ../output/java-expert \
    --bf16 \
    --gradient_checkpointing

# ===== 6. 合并模型 =====
echo "🔧 合并 LoRA 权重..."
llamafactory-cli export \
    --model_name_or_path "$MODEL" \
    --adapter_name_or_path ../output/java-expert \
    --template qwen \
    --export_dir ../java-expert-merged

cd ..

# ===== 7. 量化（可选，减小体积） =====
echo "📦 量化模型（GGUF 格式，适配 Ollama）..."
if command -v llama.cpp &> /dev/null; then
    python3 llama.cpp/convert_hf_to_gguf.py java-expert-merged --outtype q4_k_m
fi

# ===== 8. 创建 Ollama Modelfile =====
cat > Modelfile << 'OLLAMA'
FROM ./java-expert-merged

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """你是千羽蒸馏的 Java 专家助手。精通 Spring Boot、MyBatis、Redis、MySQL、并发编程。
回答简洁准确，代码示例带注释。"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
OLLAMA

echo ""
echo "================================================"
echo "  🎉 蒸馏完成！"
echo ""
echo "  模型位置: ./java-expert-merged/"
echo "  LoRA 权重: ./output/java-expert/"
echo ""
echo "  本地运行:"
echo "    ollama create java-expert -f Modelfile"
echo "    ollama run java-expert"
echo ""
echo "  测试:"
echo "    输入: 'Spring Boot 全局异常处理怎么写？'"
echo "================================================"