# transformer_model.py
import torch
import torch.nn as nn
import math
from src.embeding import TokenEmbedding, PositionalEncoding
from src.attention import MultiHeadAttention
from src.encoder_decoder import EncoderLayer, DecoderLayer

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        attn_all = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_all.append(attn)

        x = self.norm(x)
        return x, attn_all


# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        attn_all = []
        for layer in self.layers:
            x, attn = layer(x, enc_output, tgt_mask, memory_mask)
            attn_all.append(attn)

        x = self.norm(x)
        return x, attn_all


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)  # 预测下一个 token
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        enc_output, _ = self.encoder(src, src_mask)

        # 解码器
        dec_output, _ = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        # 输出词概率分布
        output = self.output_layer(dec_output)
        return output
