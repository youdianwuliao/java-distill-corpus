# Java 蒸馏语料库

> 40 条 DeepSeek-V4 生成的高质量 Java 训练数据 · 一键蒸馏脚本 · RTX 4060 实测可用

## 🎮 你的机器配置

| 部件 | 型号 |
|------|------|
| GPU | RTX 4060 (8GB) |
| 推理 | Ollama 已安装 |

## ⚡ 三步出模型

### 第一步：克隆

```bash
git clone https://github.com/youdianwuliao/java-distill-corpus.git
cd java-distill-corpus
```

### 第二步：蒸馏（30 分钟）

```bash
bash distill.sh
```

> ⚠️ 插电！笔记本训练时功耗 80-100W，电池撑不住。
> ⚠️ 笔记本垫高或架起来，风扇会全速转。

### 第三步：测试

```bash
ollama run java-expert
>>> Spring Boot 全局异常处理怎么写？
```

---

## 📊 语料覆盖

| 类别 | 条数 | 典型问题 |
|------|------|----------|
| Spring | 10 | 全局异常、限流、幂等、事务失效、AOP、事件 |
| 数据 | 10 | MyBatis 动态SQL、JPA 懒加载、Redis 穿透/击穿/雪崩、MySQL 索引/死锁 |
| 并发 | 8 | 线程池、CompletableFuture、CHM、生产者消费者、计数器 |
| 运维 | 6 | Gateway、Nacos、Docker、Git、启动优化 |
| 基础 | 6 | Stream、Record/VT、JUnit5、LRU、单例、RESTful |

---

## 🔧 distill.sh 做了什么

```
1. 检查 GPU → 安装 PyTorch + Transformers + PEFT
2. 克隆 LLaMA-Factory
3. 下载语料 + 注册数据集
4. 下载 Qwen2.5-Coder-7B + QLoRA 训练
   ├── 4-bit 量化（显存占用 ~5.5GB）
   ├── LoRA rank=8, batch=1, accumulation=8
   ├── fp16 + gradient checkpointing
   └── 3 epochs, ~25-35 分钟
5. 合并 LoRA → 完整模型
6. 导入 Ollama
```

---

## ⚙️ 参数说明

| 参数 | 值 | 原因 |
|------|-----|------|
| `load_in_4bit` | true | 8GB 显存必备，否则 OOM |
| `batch_size` | 1 | 4060 显存限制 |
| `gradient_accumulation` | 8 | 等效 batch=8 |
| `lora_rank` | 8 | 40 条语料够用，rank 太高反而过拟合 |
| `fp16` | true | 4060 支持，比 bf16 省显存 |
| `max_length` | 2048 | 语料平均 1200 token，2048 足够 |

---

## 📁 项目结构

```
java-distill-corpus/
├── README.md                  ← 本文
├── distill.sh                 ← 一键蒸馏（RTX 4060 优化版）
├── java-corpus-sharegpt.json  ← 40 条合并语料
├── batch-01~04-*.json         ← 分类语料
└── .gitignore
```

训练后新增：
```
├── output/java-expert/        ← LoRA 适配器（几 MB）
└── java-expert-merged/        ← 完整模型（7GB）
```