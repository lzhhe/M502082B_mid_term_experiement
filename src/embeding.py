# embeding.py
import os
import math
import torch
import torch.nn as nn
import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, sp_model_path, d_model=512, max_len=512):
        super().__init__()

        if not os.path.isabs(sp_model_path):
            sp_model_path = os.path.join(BASE_DIR, sp_model_path)
        if not os.path.exists(sp_model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {sp_model_path}")

        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)

        self.pos_encoding = PositionalEncoding(d_model, max_len)

    def encode_text(self, text, device=None):
        ids = [self.sp.bos_id()] + self.sp.encode(text, out_type=int) + [self.sp.eos_id()]
        ids = ids[:self.max_len]
        pad_id = self.sp.pad_id()
        if len(ids) < self.max_len:
            ids += [pad_id] * (self.max_len - len(ids))
        tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def forward(self, token_ids):
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        return self.pos_encoding(x)



if __name__ == "__main__":
    model_path = os.path.join(BASE_DIR, "bpe_tokenizer.model")
    if os.path.exists(model_path):
        print(f"Found tokenizer: {model_path}")
        embedder = TokenEmbedding(sp_model_path=model_path, d_model=256, max_len=128)
        ids = embedder.encode_text("This is a test embedding.", device="cpu")
        emb = embedder(ids)
        print(f"Embedding shape: {emb.shape}")
        print(f"Token IDs: {ids}")
    else:
        print(f"Missing tokenizer file: {model_path}")
