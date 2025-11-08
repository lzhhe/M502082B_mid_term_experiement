# M502082B大模型基础与应用 中期作业
- 刘子赫 25120399

## 实验环境及配置
使用pytorch2.1.2+cu121版本，在python3.10上运行

实验在本机上使用3060ti进行调试跑通，在4090服务器上对于超参数进行优化并进行训练
在使用NVIDIA 4090上运行一轮时间在batchsize为32时约为45秒
显存消耗约为10-12Gb左右，显卡占用约在50%左右

请在clone整个项目执行以下命令即可运行程序
$ conda create -n transformer python=3.10
$ conda activate transformer
$ cd M502082B_mid_term_experiement
$ pip install -r requirements.txt 
$ python -m src.main

## 代码结构
项目结构如下
README.md
cnn_dailymail (数据集)
├── 3.0.0
│ ├── test-00000-of-00001.parquet
│ ├── train-00000-of-00003.parquet
│ ├── train-00001-of-00003.parquet
│ ├── train-00002-of-00003.parquet
│ └── validation-00000-of-00001.parquet
requirements.txt (环境列表)
scripts
└── run.sh (运行脚本)
src
├── attention.py (注意力机制)
├── best_transformer.pth (生成的权重)
├── bpe_tokenization.py (tokenizer)
├── bpe_tokenizer.model (训练的词表模型)
├── bpe_tokenizer.vocab (词表)
├── corpus.txt (语料库)
├── dataset.py (处理数据集)
├── embeding.py (词嵌入)
├── encoder_decoder.py (编码解码器)
├── main.py (主程序)
├── test_sample_filtered.parquet (过滤好的数据集)
├── train_sample_filtered.parquet (过滤好的数据集)
├── transformer_model.py (transformer模型)
└── validation_sample_filtered.parquet (过滤好的数据集)

