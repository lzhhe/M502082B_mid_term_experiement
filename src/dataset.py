# datasetn.py
import os
import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import sentencepiece as spm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CNNDailyMailDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len=512):
        data_full_path = os.path.join(BASE_DIR, data_path)
        tokenizer_full_path = os.path.join(BASE_DIR, tokenizer_path)

        if not os.path.exists(tokenizer_full_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_full_path}")
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_full_path)

        print(f"📂 Loading dataset from: {data_full_path}")
        if not os.path.exists(data_full_path):
            raise FileNotFoundError(f"Dataset not found: {data_full_path}")

        df = pd.read_parquet(data_full_path)
        print(f"Loaded {len(df)} samples.")

        self.data = []
        self.max_len = max_len

        for _, row in df.iterrows():
            article = row.get("article")
            highlight = row.get("highlights")
            if pd.isna(article) or pd.isna(highlight):
                continue
            self.data.append((article.strip(), highlight.strip()))

    def encode_text(self, text):
        token_ids = [self.tokenizer.bos_id()] + self.tokenizer.encode(text, out_type=int) + [self.tokenizer.eos_id()]
        token_ids = token_ids[:self.max_len]
        padding = [self.tokenizer.pad_id()] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids + padding)


    def __getitem__(self, idx):
        article, highlight = self.data[idx]
        src_ids = self.encode_text(article)
        tgt_ids = self.encode_text(highlight)

        return {
            "article": src_ids,
            "decoder_input": tgt_ids[:-1],
            "decoder_output": tgt_ids[1:],
        }

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        src_batch = [x["article"] for x in batch]
        tgt_input_batch = [x["decoder_input"] for x in batch]
        tgt_output_batch = [x["decoder_output"] for x in batch]

        src_batch = torch.stack(src_batch, dim=0)
        tgt_input_batch = torch.stack(tgt_input_batch, dim=0)
        tgt_output_batch = torch.stack(tgt_output_batch, dim=0)

        return {
            "article": src_batch,
            "decoder_input": tgt_input_batch,
            "decoder_output": tgt_output_batch,
        }


if __name__ == "__main__":
    print("Downloading CNN/DailyMail dataset ...")
    ds = load_dataset("cnn_dailymail", name="3.0.0")

    # 过滤原文长度小于 1640 字符的样本
    def filter_articles_by_length(examples):
        return len(examples["article"]) <= 1640

    train_filtered = ds["train"].filter(filter_articles_by_length)
    val_filtered = ds["validation"].filter(filter_articles_by_length)
    test_filtered = ds["test"].filter(filter_articles_by_length)

    print(f"Filtered Train size: {len(train_filtered)}")
    print(f"Filtered Validation size: {len(val_filtered)}")
    print(f"Filtered Test size: {len(test_filtered)}")


    train_filtered = train_filtered.shuffle(seed=42).select(range(int(0.5 * len(train_filtered))))
    val_filtered = val_filtered.shuffle(seed=42).select(range(int(0.5 * len(val_filtered))))
    test_filtered = test_filtered.shuffle(seed=42).select(range(int(0.5 * len(test_filtered))))

    print("After sampling:")
    print(f"Train: {len(train_filtered)}  Validation: {len(val_filtered)}  Test: {len(test_filtered)}")

    train_path = os.path.join(BASE_DIR, "train_sample_filtered.parquet")
    val_path = os.path.join(BASE_DIR, "validation_sample_filtered.parquet")
    test_path = os.path.join(BASE_DIR, "test_sample_filtered.parquet")

    train_filtered.to_parquet(train_path)
    val_filtered.to_parquet(val_path)
    test_filtered.to_parquet(test_path)

    print(f"Saved filtered datasets to:\n  {train_path}\n  {val_path}\n  {test_path}")
