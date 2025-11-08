import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dimensional, head_dim):
        super(Attention, self).__init__()

        # 每个头都有自己的一组线性映射参数
        self.WQ = nn.Linear(dimensional, head_dim, bias=False)  # 查询权重矩阵
        self.WK = nn.Linear(dimensional, head_dim, bias=False)  # 键权重矩阵
        self.WV = nn.Linear(dimensional, head_dim, bias=False)  # 值的上投影

        # 初始化
        for module in [self.WQ, self.WK, self.WV]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, Q, K, V, mask=None):

        # 上投影
        Q = self.WQ(Q)  # [batch_size, seq_len, head_dim]
        K = self.WK(K)  # [batch_size, seq_len, head_dim]
        V = self.WV(V)  # [batch_size, seq_len, head_dim]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))  # [batch_size, seq_len, seq_len]

        # 掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, V)  # [batch_size, seq_len, head_dim]

        return context, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, dimensional, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dimensional = dimensional
        self.num_heads = num_heads
        self.head_dim = dimensional // num_heads

        self.heads = nn.ModuleList([
            Attention(dimensional, self.head_dim)
            for _ in range(num_heads)
        ])

        self.WO = nn.Linear(dimensional, dimensional, bias=False)
        nn.init.xavier_uniform_(self.WO.weight)

    def forward(self, Q, K, V, mask=None):
        # 每个头独立计算注意力
        head_outputs = []
        attn_list = []

        for head in self.heads:
            context, attn = head(Q, K, V, mask)
            head_outputs.append(context)
            attn_list.append(attn)

        # [seq_len, num_heads * head_dim]
        context = torch.cat(head_outputs, dim=-1)

        # 融合多头结果映射回d_model空间
        output = self.WO(context)

        # 返回输出和每个头的注意力矩阵
        return output, attn_list
