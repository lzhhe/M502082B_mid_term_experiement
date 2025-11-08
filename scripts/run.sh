#!/bin/bash

# 设置脚本遇到错误时立即退出
set -e

# 定义环境名称
ENV_NAME="transformer"
PROJECT_DIR="M502082B_mid_term_experiement"

echo "开始设置Transformer环境..."

# 创建conda环境（如果不存在）
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "环境 ${ENV_NAME} 已存在，跳过创建"
else
    echo "创建conda环境: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# 激活环境并执行后续命令
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "环境已激活: $(conda info --envs | grep '*' | awk '{print $1}')"

# 进入项目目录
if [ -d "$PROJECT_DIR" ]; then
    cd $PROJECT_DIR
    echo "已进入目录: $(pwd)"
else
    echo "错误: 目录 $PROJECT_DIR 不存在"
    exit 1
fi

# 安装依赖
echo "开始安装依赖包..."
pip install -r requirements.txt

echo "依赖安装完成"

# 运行主程序
echo "开始运行程序..."
python -m src.main

echo "程序执行完成"