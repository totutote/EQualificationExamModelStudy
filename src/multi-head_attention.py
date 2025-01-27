import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "埋め込み次元はヘッド数で割り切れる必要があります。"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # クエリ、キー、バリュー、および出力の線形変換
        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)
        self.out_lin = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size, seq_length, embed_dim = query.size()

        # 線形変換とヘッドへの分割
        Q = self.q_lin(query)  # (batch_size, seq_length, embed_dim)
        K = self.k_lin(key)
        V = self.v_lin(value)

        # ヘッド数に応じて次元を変換
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # 次元を入れ替え (batch_size, num_heads, seq_length, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # スケールド・ドットプロダクト注意
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

        # ヘッドを結合
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_length, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_length, embed_dim)  # (batch_size, seq_length, embed_dim)

        # 最終的な線形変換
        output = self.out_lin(attn_output)  # (batch_size, seq_length, embed_dim)

        return output, attn_weights

# パラメータ設定
batch_size = 2
seq_length = 5
embed_dim = 16
num_heads = 4

# ランダムな入力データ
x = torch.randn(batch_size, seq_length, embed_dim)

# マルチヘッド注意機構のインスタンス化
multihead_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

# 前方伝搬
attn_output, attn_weights = multihead_attn(x, x, x)

print("Attention output shape:", attn_output.shape)
print("Attention output:", attn_output)
print("Attention weights shape:", attn_weights.shape)
print("Attention weights:", attn_weights)
