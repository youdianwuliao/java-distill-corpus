#!/bin/bash
# 清理脚本 — 删除蒸馏产生的所有文件和虚拟环境
set -e

echo "🧹 清理 Java 蒸馏环境"
echo ""
echo "以下将被删除:"
echo "  venv/            Python 虚拟环境"
echo "  LLaMA-Factory/   训练框架"
echo "  output/          LoRA 权重"
echo "  java-expert-merged/ 合并模型"
echo "  Modelfile        Ollama 配置"
echo "  ~/.cache/huggingface/ 模型缓存（可选）"
echo ""

read -p "确认删除? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "取消"
    exit 0
fi

rm -rf venv
rm -rf LLaMA-Factory
rm -rf output
rm -rf java-expert-merged
rm -f Modelfile

# 模型缓存很大（~14GB），单独确认
read -p "同时删除 HuggingFace 模型缓存 (~14GB)? (yes/no): " CACHE_CONFIRM
if [ "$CACHE_CONFIRM" = "yes" ]; then
    rm -rf ~/.cache/huggingface/
    rm -rf ~/.cache/torch/
    echo "✅ 缓存已删除"
fi

# 删除 Ollama 模型
if ollama list 2>/dev/null | grep -q java-expert; then
    read -p "同时删除 Ollama 中的 java-expert 模型? (yes/no): " OLLAMA_CONFIRM
    if [ "$OLLAMA_CONFIRM" = "yes" ]; then
        ollama rm java-expert 2>/dev/null
        echo "✅ Ollama 模型已删除"
    fi
fi

echo ""
echo "✅ 清理完成"
echo ""
echo "重新蒸馏: bash distill.sh"