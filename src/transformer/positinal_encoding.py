import torch
import numpy as np
import matplotlib.pyplot as plt

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 埋め込みベクトルの次元数
        max_len: 最大系列長
        """
        super(PositionalEncoding, self).__init__()

        # 各位置と各次元のポジショナルエンコーディングを計算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 偶数インデックスにはsinを適用
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 奇数インデックスにはcosを適用
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 次元を追加してバッチ処理に対応 [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # モデルのパラメータとして登録するが、勾配計算は不要
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]の入力テンソル
        """
        # 入力テンソルに対応する位置のエンコーディングを加算
        return x + self.pe[:, :x.size(1), :]

def visualize_positional_encoding(d_model=128, max_len=100):
    """ポジショナルエンコーディングの可視化"""
    import platform
    # OSに応じた日本語対応フォントに切り替え
    if platform.system() == "Windows":
        plt.rcParams["font.family"] = "Meiryo"          # Windows: 日本語対応フォント "Meiryo"
    elif platform.system() == "Darwin":
        plt.rcParams["font.family"] = "Hiragino Sans"     # macOS: 日本語対応フォント "Hiragino Sans"

    # ポジショナルエンコーディングのインスタンスを作成
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
    
    # エンコーディング行列を取得
    pe = pos_encoder.pe.squeeze().numpy()
    
    plt.figure(figsize=(15, 8))
    
    # ヒートマップとして可視化
    plt.subplot(2, 1, 1)
    plt.imshow(pe, aspect='auto', cmap='viridis')
    plt.xlabel('次元 (d_model)')
    plt.ylabel('位置 (position)')
    plt.title('ポジショナルエンコーディング全体の可視化')
    plt.colorbar()
    
    # 特定の位置でのエンコーディングをプロット
    plt.subplot(2, 1, 2)
    positions = [0, 10, 20, 50]
    for pos in positions:
        plt.plot(pe[pos, :], label=f'position={pos}')
    plt.legend()
    plt.xlabel('次元 (d_model)')
    plt.ylabel('エンコーディング値')
    plt.title('異なる位置でのエンコーディング')
    
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png')
    plt.show()
    
    # 特定の次元に対する位置の関数としてのエンコーディング
    plt.figure(figsize=(15, 6))
    dimensions = [0, 1, 4, d_model//2, d_model-2, d_model-1]
    for dim in dimensions:
        plt.plot(pe[:, dim], label=f'dim={dim}')
    plt.legend()
    plt.xlabel('位置 (position)')
    plt.ylabel('エンコーディング値')
    plt.title('異なる次元でのエンコーディングの周期性')
    plt.grid(True)
    
    plt.savefig('positional_encoding_periodicity.png')
    plt.show()

# 実行例
if __name__ == "__main__":
    # ポジショナルエンコーディングを可視化
    visualize_positional_encoding(d_model=64, max_len=200)
    
    # ポジショナルエンコーディングの基本的な使用例
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    # ランダムな埋め込みベクトルを生成
    embeddings = torch.rand(batch_size, seq_len, d_model)
    
    # ポジショナルエンコーディングを適用
    pos_encoder = PositionalEncoding(d_model)
    output = pos_encoder(embeddings)
    
    print(f"入力形状: {embeddings.shape}")
    print(f"出力形状: {output.shape}")
    print("位置情報が埋め込まれました！")
