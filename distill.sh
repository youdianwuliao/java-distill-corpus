#!/usr/bin/env bash
# Java 蒸馏 — Deepin/RTX 4060 版
# 用法: bash distill.sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

die() { echo -e "${RED}$1${NC}"; exit 1; }
step() { echo ""; echo -e "${GREEN}===== $1 =====${NC}"; }

step "0/5 环境检查"

nvidia-smi &>/dev/null || die "没检测到 NVIDIA 驱动"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "CUDA: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')"
echo "Python: $(python3 --version)"
echo "磁盘: $(df -h . | tail -1 | awk '{print $4}') 可用"

# 系统依赖
dpkg -l build-essential &>/dev/null 2>&1 || sudo apt install -y build-essential python3-pip python3-venv git curl
python3 -m pip --version &>/dev/null 2>&1 || sudo apt install -y python3-pip

# 虚拟环境
if [ ! -d "venv" ]; then python3 -m venv venv; fi
source venv/bin/activate

# 镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$(pwd)/.cache/huggingface"
export TORCH_HOME="$(pwd)/.cache/torch"

# 跳过已安装的
step "1/5 安装 Python 包"
pip install --upgrade pip -q 2>/dev/null
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers datasets accelerate peft bitsandbytes scipy -q
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

step "2/5 准备 LLaMA-Factory + 语料"
[ ! -d "LLaMA-Factory" ] && git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch]" -q 2>/dev/null && cd ..

[ ! -f "java-corpus-sharegpt.json" ] && curl -sL "https://raw.githubusercontent.com/youdianwuliao/java-distill-corpus/main/java-corpus-sharegpt.json" -o java-corpus-sharegpt.json
cp java-corpus-sharegpt.json LLaMA-Factory/data/

python3 -c "
import json
with open('LLaMA-Factory/data/dataset_info.json') as f:
    info = json.load(f)
info['java_corpus'] = {
    'file_name': 'java-corpus-sharegpt.json',
    'formatting': 'sharegpt',
    'columns': {'messages': 'conversations'}
}
with open('LLaMA-Factory/data/dataset_info.json', 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print('数据集已注册')
"

step "3/5 训练（约25-35分钟）"
echo -e "${YELLOW}⚠️ 插电+垫高！风扇起飞正常${NC}"
echo ""

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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 1000 \
    --max_grad_norm 1.0 \
    --output_dir ../output/java-expert \
    --fp16 \
    --gradient_checkpointing \
    --max_length 2048

TRAIN_RESULT=$?
cd ..

if [ $TRAIN_RESULT -ne 0 ]; then
    echo ""
    echo -e "${RED}训练失败 (exit code: $TRAIN_RESULT)${NC}"
    echo "把上面的完整日志发给我"
    deactivate
    exit 1
fi

# 检查产出
if [ ! -d "output/java-expert" ] || [ -z "$(ls -A output/java-expert/)" ]; then
    die "训练完成但 output/java-expert/ 为空 — 把训练日志发给我"
fi
echo "✅ 训练产出: $(ls output/java-expert/)"

step "4/5 合并模型"
cd LLaMA-Factory

ADAPTER="../output/java-expert"
# 如果有 checkpoint 子目录就用最新的
CHECKPOINT=$(ls -d ../output/java-expert/checkpoint-* 2>/dev/null | sort -V | tail -1)
[ -n "$CHECKPOINT" ] && ADAPTER="$CHECKPOINT"
echo "适配器路径: $ADAPTER"

llamafactory-cli export \
    --model_name_or_path "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --adapter_name_or_path "$ADAPTER" \
    --template qwen \
    --export_dir ../java-expert-merged \
    --export_size 2 \
    --export_legacy_format false || die "合并失败"

cd ..

step "5/5 导入 Ollama"
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
echo "  ollama run java-expert"
echo "  > Spring Boot 全局异常处理怎么写？"
echo "================================================"