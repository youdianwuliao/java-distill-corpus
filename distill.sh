#!/bin/bash
# ================================================================
# Java 蒸馏 — RTX 4060 笔记本专用（8GB 显存）
#
# 用法:
#   bash distill.sh
#
# 前提:
#   - Windows/Linux + RTX 4060 (8GB)
#   - 已安装 CUDA 12.x 驱动
#   - 已安装 Ollama
#   - 50GB 磁盘空间
# ================================================================
set -e

echo "🪶 千羽 Java 蒸馏 — RTX 4060 版"

# ===== 0. 环境检查 =====
echo ""
echo "===== 环境检查 ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠️ 未检测到 NVIDIA GPU"
echo "Python: $(python3 --version)"
echo "磁盘: $(df -h . | tail -1 | awk '{print $4}') 可用"

# ===== 1. 安装依赖 =====
echo ""
echo "===== 1/6 安装依赖 ====="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>&1 | tail -1
pip install transformers datasets accelerate peft bitsandbytes scipy -q 2>&1 | tail -1

# ===== 2. 拉 LLaMA-Factory =====
echo ""
echo "===== 2/6 获取 LLaMA-Factory ====="
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e ".[torch]" -q 2>&1 | tail -1
cd ..

# ===== 3. 准备语料 =====
echo ""
echo "===== 3/6 准备语料 ====="
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
echo "===== 4/6 开始训练（约 25-35 分钟）====="
echo "⚠️ 训练期间笔记本会发热，插电 + 垫高散热！"

cd LLaMA-Factory

llamafactory-cli train \
    --model_name_or_path "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --dataset java_corpus \
    --template qwen \
    --finetuning_type lora \
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
    --load_in_4bit \
    --bnb_4bit_compute_dtype float16 \
    --bnb_4bit_quant_type nf4 \
    --max_length 2048

cd ..

# ===== 5. 合并模型 =====
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
echo "===== 6/6 导入 Ollama ====="

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

echo ""
echo "================================================"
echo "  🎉 蒸馏完成！"
echo ""
echo "  测试:"
echo "    ollama run java-expert"
echo "    > Spring Boot 全局异常处理怎么写？"
echo ""
echo "  输出文件:"
echo "    完整模型: ./java-expert-merged/"
echo "    LoRA权重: ./output/java-expert/"
echo "    Ollama:   ollama list | grep java-expert"
echo "================================================"