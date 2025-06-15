# dqn-cartpoleプロジェクト

このプロジェクトは、Gymnasium環境であるCartPoleを使用して、DQN（Deep Q-Network）エージェントを訓練し、学習済みモデルを使用して環境を操作することを目的としています。

## プロジェクト構成

```
dqn-cartpole
├── src
│   ├── train.py        # DQNエージェントを訓練するためのエントリーポイント
│   ├── run_model.py    # 学習済みのDQNモデルを使用して環境を操作
│   └── model.py        # DQNエージェントのモデル定義
├── environment.yml      # Conda環境の設定ファイル
└── README.md            # プロジェクトの概要と使用方法
```

## セットアップ手順

1. リポジトリをクローンします。
   ```
   git clone <repository-url>
   cd dqn-cartpole
   ```

2. 必要なパッケージをインストールします。
   ```
   pip install -r requirements.txt
   ```

   または、Conda環境を使用する場合は以下のコマンドを実行します。
   ```
   conda env create -f environment.yml
   conda activate gymnasium
   ```

3. ffmpegをインストールします（動画記録のために必要です）。
   ```
   brew install ffmpeg
   ```

## 使用方法

### DQNエージェントの訓練

`src/train.py`を実行して、DQNエージェントを訓練します。

```
python src/train.py
```

### 学習済みモデルの実行

訓練済みのモデルを使用してCartPole環境を操作するには、`src/run_model.py`を実行します。

```
python src/run_model.py
```

## 注意事項

- このプロジェクトはPyTorchを使用して実装されています。
- 環境の動作にはGymnasiumが必要です。
- このプロジェクトはmacOS 15環境でのみ確認されています。
