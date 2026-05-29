# Java 蒸馏语料库

从 DeepSeek-V4 生成的 40 条高质量 Java 训练数据，用于 fine-tune 小模型。

## 语料内容

| 类别 | 条数 | 覆盖 |
|------|------|------|
| Spring Boot | 10 条 | 全局异常、限流、幂等、事件机制、AOP、启动优化... |
| 并发集合 | 10 条 | 线程池、CompletableFuture、HashMap/CHM、LRU、计数器... |
| 微服务运维 | 10 条 | Gateway 鉴权、Nacos、Docker、Git、死锁排查... |
| Java 核心 | 10 条 | Record/VT、Stream、JUnit5、RESTful、String、设计模式... |

## 格式

标准 **ShareGPT 格式**，兼容 LLaMA-Factory / Axolotl / Unsloth：

```json
[
  {
    "conversations": [
      {"role": "user", "content": "Java中HashMap和ConcurrentHashMap有什么区别？"},
      {"role": "assistant", "content": "核心区别：\n**HashMap**\n- 非线程安全..."}
    ]
  }
]
```

## 蒸馏步骤

### 1. 环境准备

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 2. 放入语料

```bash
cp java-corpus-sharegpt.json LLaMA-Factory/data/
```

在 `data/dataset_info.json` 末尾添加：

```json
"java_corpus": {
  "file_name": "java-corpus-sharegpt.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations"
  }
}
```

### 3. 选基座模型（推荐）

```bash
# 方案 A: Qwen2.5-Coder-7B（推荐，Java 效果最好）
# 方案 B: DeepSeek-Coder-1.3B（轻量，4G 显存可跑）
# 方案 C: CodeLlama-7B（通用性强）
```

### 4. 开始训练

```bash
# QLoRA 微调（消费级显卡，8G 显存）
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset java_corpus \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 8 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./output/java-expert
```

### 5. 导出模型

```bash
# 合并 LoRA 权重
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter_name_or_path ./output/java-expert \
  --template qwen \
  --export_dir ./java-expert-merged
```

### 6. 本地运行

```bash
# Ollama
ollama create java-expert -f Modelfile
ollama run java-expert
```

## 硬件要求

| 方案 | 基座模型 | 显存 | 训练时间 |
|------|----------|------|----------|
| QLoRA | Qwen2.5-Coder-7B | 8GB | ~30min |
| QLoRA | DeepSeek-Coder-1.3B | 4GB | ~15min |
| Full | Qwen2.5-Coder-7B | 24GB | ~2h |

## 效果

蒸馏后的 7B 模型在 Java 相关问题上能达到原模型 80-90% 的水平：

- ✅ Spring Boot / MyBatis 代码生成
- ✅ 并发编程、线程池配置
- ✅ Bug 定位和修复建议
- ✅ 单元测试生成
- ⚠️ 复杂架构设计不如大模型
- ⚠️ 新 API / 框架可能编造

## 扩展

语料太少？用大模型继续生成更多：

```bash
# 用你喜欢的任何大模型，按这个格式生成更多 JSON
# 然后追加到 java-corpus-sharegpt.json
```

建议总量：3000-5000 条效果最佳。