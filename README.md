# Java 蒸馏语料库 — 千羽 🪶

> 从 DeepSeek-V4 生成的 60 条 Java 训练数据 → 1.5B QLoRA 微调 → Ollama 本地运行

## 一步到位

```bash
git clone https://github.com/youdianwuliao/java-distill-corpus.git
cd java-distill-corpus
bash distill.sh
```

脚本自动完成：环境检查 → 下载模型 + 语料 → QLoRA 训练 → GGUF 转换 → Ollama 导入

**一键出模型** 🪶

---

## 📊 语料

| 批次 | 覆盖 | 条数 |
|------|------|------|
| batch-01 | Spring Boot（全局异常、限流、幂等、事件...） | 10 |
| batch-02 | 并发/数据（线程池、CompletableFuture、Redis...） | 10 |
| batch-03 | 微服务（Gateway、Nacos、Docker、Git...） | 10 |
| batch-04 | Java 核心（17/21新特性、LRU、设计模式...） | 10 |
| batch-05 | Spring 进阶（多数据源、延迟队列、AOP日志...） | 10 |
| batch-06 | 工具（枚举策略、Arthas、Linux、EasyExcel...） | 10 |

格式：ShareGPT（`from/value`），LLaMA-Factory / 直接训练两用

---

## 🔧 手动步骤（如果不想全自动）

### 1. 环境

```bash
# Deepin / Ubuntu
sudo apt install -y build-essential python3-pip python3-venv git curl

# 虚拟环境
python3 -m venv venv
source venv/bin/activate

# PyTorch + 依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate
```

### 2. 训练

```bash
python3 << 'EOF'
# transformers + peft 直接训练，不依赖 LLaMA-Factory
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from peft import LoraConfig, get_peft_model
import json

# 加载 1.5B 模型 + 语料 → QLoRA 训练 3 轮
# 详见 distill.sh 第 70-130 行
EOF
```

### 3. 合并 + GGUF

```bash
# 获取 llama.cpp（Gitee 镜像，GitHub 可能连不上）
git clone --depth 1 https://gitee.com/mirrors/llama.cpp.git
cd llama.cpp && make -j4 && cd ..

# 转 GGUF
python3 llama.cpp/convert_hf_to_gguf.py \
    java-expert-merged --outtype f16 --outfile java-expert.gguf
```

### 4. Ollama

```bash
ollama create java-expert -f Modelfile
ollama run java-expert

# 测试
>>> 写个冒泡排序
```

---

## 🚀 如果重跑

```bash
bash clean.sh && bash distill.sh
# 等 ~20 分钟（1.5B, 3 epochs, RTX 4060）
# 自动出 ollama model: java-expert:latest
```

---

## ⚙️ 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen2.5-Coder-1.5B-Instruct | 1.5B, 适合 8GB 显存 |
| 训练方法 | QLoRA (4-bit) | 省显存 |
| LoRA rank | 8 | 60 条语料够用 |
| batch size | 1, accum 8 | 等效 batch=8 |
| epochs | 3 | 不过拟合 |
| fp16 | true | RTX 4060 支持 |
| max_length | 1024 | 语料平均 800 token |

## 📁 项目结构

```
java-distill-corpus/
├── distill.sh              ← 一键执行（从零到 Ollama）
├── clean.sh                ← 清理（删 venv + 模型 + 缓存）
├── java-corpus-sharegpt.json  ← 60 条合并语料
├── batch-01~06-*.json      ← 分批次原始数据
├── .gitignore
├── README.md               ← 本文
├── venv/                   ← 虚拟环境（distill.sh 自动创建）
├── .cache/huggingface/     ← 模型下载缓存（~3GB）
├── output/java-expert/     ← LoRA 适配器
├── java-expert-merged/     ← 合并后模型
├── java-expert.gguf        ← GGUF 格式（Ollama 原生）
├── llama.cpp/              ← 转换工具
└── Modelfile               ← Ollama 配置
```

## 🧪 测试

```bash
ollama run java-expert
>>> Spring Boot 全局异常处理怎么写？
>>> HashMap 和 ConcurrentHashMap 的区别？
>>> 写一个单例模式
```

---

_千羽出品 🪶_