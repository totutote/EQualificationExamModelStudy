import gymnasium as gym
import torch
import numpy as np
from model import DQN
from gymnasium.wrappers import RecordVideo

def run_model(model_path):
    # デバイスの設定
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # CartPole環境の初期化
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RecordVideo(env, './video_run', episode_trigger=lambda episode_id: True)
    env._max_episode_steps = 1000
    state, _ = env.reset()

    # 学習済みモデルのロード
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size=state_size, action_size=action_size, hidden_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)) 

    done = False
    total_reward = 0

    while not done:
        # 状態をテンソルに変換
        state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        
        # 行動の選択
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        # 環境で行動を実行
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # 環境の描画
        env.render()

    print(f'Total Reward: {total_reward}')
    env.close()

if __name__ == "__main__":
    run_model('dqn_cartpole.pth')
