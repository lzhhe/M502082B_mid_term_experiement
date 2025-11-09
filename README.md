# M502082B 大模型基础与应用 · 中期作业  
**姓名：刘子赫**  **学号：25120399**

---

## 实验环境与配置

- **操作系统与依赖环境**  
  - Python：`3.10`  
  - PyTorch：`2.1.2 + CUDA 12.1`

- **硬件配置**  
  - 本地调试环境：NVIDIA **RTX 3060 Ti**  
  - 训练优化环境：NVIDIA **RTX 4090**，在**Batch Size = 32**时单轮训练耗时约**45 秒**，显存占用约**10–12 GB**，显卡利用率约 **50%**。

---

## 快速开始

请在克隆项目之后执行以下命令即可运行程序

```bash
# 创建并激活环境
conda create -n transformer python=3.10
conda activate transformer

# 进入项目目录
cd M502082B_mid_term_experiement

# 安装依赖
pip install -r requirements.txt 

# 运行主程序
python -m src.main
```


## 项目结构
```bash
M502082B_mid_term_experiement/
│
├── README.md                      # 项目说明文件
├── requirements.txt               # 环境依赖列表
│
├── cnn_dailymail/                 # 数据集目录
│   └── 3.0.0/
│       ├── train-00000-of-00003.parquet
│       ├── train-00001-of-00003.parquet
│       ├── train-00002-of-00003.parquet
│       ├── validation-00000-of-00001.parquet
│       └── test-00000-of-00001.parquet
│
├── scripts/
│   └── run.sh                     # 运行脚本
│
└── src/                           # 源代码目录
    ├── attention.py               # 注意力机制实现
    ├── best_transformer.pth       # 最优模型权重
    ├── bpe_tokenization.py        # BPE分词器代码
    ├── bpe_tokenizer.model        # 训练好的分词模型
    ├── bpe_tokenizer.vocab        # 词表文件
    ├── corpus.txt                 # 语料库
    ├── dataset.py                 # 数据加载与处理
    ├── embeding.py                # 词嵌入层
    ├── encoder_decoder.py         # 编码器-解码器结构
    ├── main.py                    # 主程序入口
    ├── transformer_model.py       # Transformer 模型定义
    ├── train_sample_filtered.parquet
    ├── validation_sample_filtered.parquet
    └── test_sample_filtered.parquet
```
