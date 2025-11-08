import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ======================================================
# 📁 跨平台路径根：脚本所在目录
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================
# 模型定义 (与训练时完全一致)
# ==============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Attention(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.WQ = nn.Linear(d_model, head_dim, bias=False)
        self.WK = nn.Linear(d_model, head_dim, bias=False)
        self.V_up = nn.Linear(d_model, head_dim, bias=False)  # 关键：改为 V_up
        for m in [self.WQ, self.WK, self.V_up]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, Q, K, V, mask=None):
        Q = self.WQ(Q)
        K = self.WK(K)
        V_up = self.V_up(V)  # 关键：改为 V_up

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V_up)  # 关键：使用 V_up
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.heads = nn.ModuleList([Attention(d_model, self.head_dim) for _ in range(num_heads)])
        self.V_down = nn.Linear(d_model, d_model, bias=False)  # 关键：改为 V_down
        nn.init.xavier_uniform_(self.V_down.weight)

    def forward(self, Q, K, V, mask=None):
        head_outputs = []
        for head in self.heads:
            context, _ = head(Q, K, V, mask)
            head_outputs.append(context)
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.V_down(concatenated)  # 关键：使用 V_down
        return output, None

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x, None

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        cross_attn_out, attn_weights = self.cross_attn(x, enc_output, enc_output, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x, attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x, _ = layer(x, mask)
        x = self.norm(x)
        return x, None

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x, _ = layer(x, enc_output, tgt_mask, memory_mask)
        x = self.norm(x)
        return x, None

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output, _ = self.encoder(src, src_mask)
        dec_output, _ = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        output = self.output_layer(dec_output)
        return output

# ==============================
# 修复后的摘要生成器类
# ==============================

class TextSummarizer:
    def __init__(self, model_path, tokenizer_path, max_len=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        # 规范化路径
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.join(BASE_DIR, tokenizer_path)

        # 加载tokenizer
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"❌ 找不到tokenizer模型: {tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.vocab_size = self.tokenizer.get_piece_size()

        print(f"✅ Tokenizer加载成功，词表大小: {self.vocab_size}")
        print(f"特殊标记 - PAD: {self.tokenizer.pad_id()}, BOS: {self.tokenizer.bos_id()}, EOS: {self.tokenizer.eos_id()}")

        # 初始化模型 - 必须与训练时完全一致！
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            d_model=128,      # 根据你的训练代码
            num_heads=2,      # 根据你的训练代码
            num_layers=2,     # 根据你的训练代码
            d_ff=512          # 根据你的训练代码
        ).to(self.device)

        # 加载权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型权重文件: {model_path}")
        
        print(f"📂 加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 关键：加载权重并处理可能的键不匹配
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 尝试加载权重
        try:
            self.model.load_state_dict(state_dict)
            print("✅ 模型权重加载成功!")
        except Exception as e:
            print(f"⚠️ 直接加载失败，尝试处理键不匹配: {e}")
            # 处理键不匹配
            model_dict = self.model.state_dict()
            
            # 创建映射关系
            key_mapping = {}
            for key in state_dict.keys():
                if 'WV' in key:
                    new_key = key.replace('WV', 'V_up')
                elif 'WO' in key:
                    new_key = key.replace('WO', 'V_down')
                else:
                    new_key = key
                key_mapping[new_key] = key
            
            # 创建新的state_dict
            new_state_dict = {}
            for model_key in model_dict.keys():
                if model_key in key_mapping:
                    # 直接映射
                    new_state_dict[model_key] = state_dict[key_mapping[model_key]]
                elif model_key in state_dict:
                    # 直接使用
                    new_state_dict[model_key] = state_dict[model_key]
                else:
                    # 使用随机初始化
                    print(f"⚠️ 找不到参数: {model_key}，使用随机初始化")
                    new_state_dict[model_key] = model_dict[model_key]
            
            # 加载处理后的权重
            self.model.load_state_dict(new_state_dict)
            print("✅ 处理后权重加载成功!")
        
        self.model.eval()
        print("✅ 文本摘要器初始化完成!")

    def generate_square_subsequent_mask(self, sz, device=None):
        """生成因果mask"""
        if device is None:
            device = self.device
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def summarize(self, text, max_length=80, temperature=1.0, top_k=50):
        """生成文本摘要"""
        # 编码输入文本
        src_ids = self.tokenizer.encode(text, out_type=int)
        src_ids = src_ids[:self.max_len]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # 初始化解码器输入
        tgt_ids = torch.tensor([[self.tokenizer.bos_id()]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for step in range(max_length):
                tgt_mask = self.generate_square_subsequent_mask(tgt_ids.size(1), self.device)
                output = self.model(src_tensor, tgt_ids, tgt_mask=tgt_mask)
                
                # 获取下一个token的logits
                next_token_logits = output[:, -1, :]
                
                # 应用温度调节
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k筛选
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # 采样
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪婪解码
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 添加到序列
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
                
                # 重复检测
                if step > 10:
                    recent_tokens = tgt_ids[0][-5:].tolist()
                    if len(set(recent_tokens)) == 1:
                        print("⚠️ 检测到重复，提前结束生成")
                        break
                
                # EOS停止
                if next_token.item() == self.tokenizer.eos_id():
                    break

        # 解码生成结果
        generated_ids = tgt_ids[0].tolist()[1:]  # 去掉BOS
        if self.tokenizer.eos_id() in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(self.tokenizer.eos_id())]
        
        return self.tokenizer.decode(generated_ids)

# ==============================
# 主程序
# ==============================

def main():
    try:
        summarizer = TextSummarizer(
            model_path="best_transformer.pth",
            tokenizer_path="bpe_tokenizer.model"
        )

        print("\n" + "=" * 60)
        print("📝 文本摘要生成器")
        print("=" * 60)
        print("输入 'quit' 退出程序")
        print("=" * 60)

        while True:
            print("\n" + "-" * 40)
            text = input("\n请输入要摘要的文本: ").strip()
            
            if text.lower() in {"quit", "退出", "exit"}:
                print("👋 再见!")
                break
                
            if not text:
                print("⚠️ 请输入有效文本")
                continue

            print("\n⏳ 正在生成摘要...")
            try:
                # 先用贪婪解码测试
                summary_greedy = summarizer.summarize(text, temperature=0, top_k=0)
                print(f"\n📄 原文: {text}")
                print(f"📝 贪婪解码: {summary_greedy}")
                
                # 再用采样生成
                summary_sampled = summarizer.summarize(text, temperature=0.8, top_k=30)
                print(f"📝 采样生成: {summary_sampled}")
                
            except Exception as e:
                print(f"❌ 生成摘要时出错: {e}")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == "__main__":
    main()