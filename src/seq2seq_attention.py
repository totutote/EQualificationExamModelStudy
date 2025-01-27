import random
import torch
import torch.nn as nn
import torch.optim as optim

# エンコーダーの定義
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers)
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

# Attention機構の定義
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden_repeat = hidden[-1].unsqueeze(1).repeat(1, src_len, 1).permute(1, 0, 2)
        combined = torch.cat((hidden_repeat, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(combined))
        v = self.v.repeat(encoder_outputs.shape[1], 1).unsqueeze(1)
        energy_permute = energy.permute(1, 2, 0)
        attention = torch.bmm(v, energy_permute).squeeze(1)
        return torch.softmax(attention, dim=1)

# デコーダーの定義
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs.permute(1, 0, 2))
        rnn_input = torch.cat((embedded, weighted.permute(1, 0, 2)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(1)), dim=1))
        return prediction, hidden, cell

# Seq2Seqモデルの定義
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1
        return outputs

# 定数の定義
SRC_VOCAB_SIZE = 10 # 学習文章の語彙数
TRG_VOCAB_SIZE = 10 # 目的文章の語彙数
EMB_DIM = 8 # 単語埋め込みの次元数
HID_DIM = 16 # LSTMの隠れ層の次元数
N_LAYERS = 1 # LSTMの層数
SRC_SEQ_LEN = 15  # 学習文章中の単語数
TRG_SEQ_LEN = 20  # 目的文章中の単語数
BATCH_SIZE = 2   # 文章数

enc = Encoder(SRC_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS)
attn = Attention(HID_DIM)
dec = Decoder(TRG_VOCAB_SIZE, EMB_DIM, HID_DIM, N_LAYERS, attn)

device = torch.device('mps' if torch.mps.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)

# 仮の入力データ
src = torch.randint(0, SRC_VOCAB_SIZE, (SRC_SEQ_LEN, BATCH_SIZE)).to(device)
trg = torch.randint(0, TRG_VOCAB_SIZE, (TRG_SEQ_LEN, BATCH_SIZE)).to(device)

# モデルのテスト
outputs = model(src, trg)
print(outputs.shape)  # 出力の形状を確認