#!/usr/bin/env bash
# Java 蒸馏 — Deepin/RTX 4060 版 v6
# 纯 Python 训练，不依赖 LLaMA-Factory

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🪶 千羽 Java 蒸馏 v6 — 纯 Python 版${NC}"

# ===== 环境 =====
echo ""
echo "===== 环境检查 ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo -e "${RED}无 GPU${NC}"; exit 1; }
echo "Python: $(python3 --version)"
echo "磁盘: $(df -h . | tail -1 | awk '{print $4}')"

# 系统依赖
dpkg -l build-essential &>/dev/null 2>&1 || sudo apt install -y build-essential python3-pip python3-venv git curl
python3 -m pip --version &>/dev/null 2>&1 || sudo apt install -y python3-pip

# venv
if [ ! -d "venv" ]; then python3 -m venv venv; fi
source venv/bin/activate

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$(pwd)/.cache/huggingface"

# ===== 安装 =====
echo ""
echo "===== 安装依赖 ====="
pip install --upgrade pip -q 2>/dev/null
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers datasets peft accelerate -q 2>/dev/null
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# ===== 语料 =====
echo ""
echo "===== 准备语料 ====="
if [ ! -f "java-corpus-sharegpt.json" ]; then
    curl -sL "https://raw.githubusercontent.com/youdianwuliao/java-distill-corpus/main/java-corpus-sharegpt.json" \
        -o java-corpus-sharegpt.json
fi
python3 -c "
import json
data = json.load(open('java-corpus-sharegpt.json'))
print(f'语料: {len(data)} 条')
"

# ===== 训练 =====
echo ""
echo "===== 开始训练 ====="
echo -e "${YELLOW}⚠️ 插电+垫高！${NC}"

python3 << 'TRAIN'
import torch, json, os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "output/java-expert"

# 1. 加载语料
print("加载语料...")
raw = json.load(open("java-corpus-sharegpt.json"))
texts = []
for item in raw:
    conv = item["conversations"]
    user_msg = ""
    assistant_msg = ""
    for c in conv:
        role = c.get("from", c.get("role", ""))
        content = c.get("value", c.get("content", ""))
        if role in ("human", "user"):
            user_msg = content
        elif role in ("gpt", "assistant"):
            assistant_msg = content
    if user_msg and assistant_msg:
        # Qwen chat format
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        texts.append(text)

print(f"训练样本: {len(texts)} 条")
dataset = Dataset.from_dict({"text": texts})

# 2. 加载模型和 tokenizer
print(f"加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.enable_input_require_grads()

# 3. LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Tokenize
def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding=False
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("Tokenizing...")
dataset = dataset.map(tokenize, remove_columns=["text"])

# 5. 训练
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=3,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
    save_strategy="steps",
    logging_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("\n🔥 开始训练...\n")
trainer.train()

# 6. 保存
print("\n保存模型...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 训练完成！模型保存到 {OUTPUT_DIR}")
TRAIN

RET=$?
if [ $RET -ne 0 ]; then
    echo ""
    echo -e "${RED}训练失败 (exit: $RET)${NC}"
    deactivate
    exit 1
fi

# ===== 检查 =====
if [ ! -f "output/java-expert/adapter_config.json" ]; then
    echo -e "${RED}adapter_config.json 不存在${NC}"
    find output/ -type f 2>/dev/null | head -10
    deactivate
    exit 1
fi
echo "✅ adapter_config.json 确认"

# ===== 合并 =====
echo ""
echo "===== 合并模型 ====="
python3 << 'MERGE'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "output/java-expert"
output_path = "java-expert-merged"

print("加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

print("加载 LoRA...")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

print("保存...")
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)
print(f"✅ 合并完成: {output_path}")
MERGE

# ===== Ollama =====
echo ""
echo "===== 导入 Ollama ====="
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
echo -e "${GREEN}🎉 完成！ollama run java-expert${NC}"