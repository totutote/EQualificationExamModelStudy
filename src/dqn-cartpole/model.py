import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.model = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_model = DQN(state_size, action_size, hidden_size).to(self.device)
        self.update_target_network()
        self.update_target_every = 1000
        self.step_count = 0
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.append((state, action, reward, next_state, terminated, truncated))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.clone().detach().to(self.device).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminateds, truncateds = zip(*minibatch)
        
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        terminateds = torch.tensor(terminateds, dtype=torch.bool, device=self.device)
        truncateds = torch.tensor(truncateds, dtype=torch.bool, device=self.device)

        # 現在の状態からQ値を取得
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        model_q = self.model(states)
        current_q = model_q.gather(1, actions)

        # ターゲットネットワークから次の状態の最大Q値を取得
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]

        # ターゲットQ値の計算
        target_q = rewards + (self.gamma * max_next_q * (~terminateds))

        # 損失の計算とバックプロパゲーション
        loss = self.criterion(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # イプシロンの減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # ターゲットネットワークの更新
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()
