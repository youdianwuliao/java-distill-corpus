#!/bin/bash
# 清理脚本 — 删除蒸馏产生的所有文件
# 所有文件都在 java-distill-corpus/ 目录下，删目录就干净了

echo "🧹 清理 Java 蒸馏环境"
echo ""

# 检查并删除
for dir in venv LLaMA-Factory output java-expert-merged .cache; do
    if [ -d "$dir" ]; then
        echo "  删除 $dir/"
        rm -rf "$dir"
    fi
done
[ -f Modelfile ] && rm -f Modelfile && echo "  删除 Modelfile"

# Ollama 模型
if ollama list 2>/dev/null | grep -q java-expert; then
    ollama rm java-expert 2>/dev/null
    echo "  删除 Ollama java-expert"
fi

echo ""
echo "✅ 清理完成。重新蒸馏: bash distill.sh"