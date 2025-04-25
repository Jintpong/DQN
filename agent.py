import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import pandas as pd
import wandb
import copy

# Initialize Weights & Biases
wandb.init(project="CW2-DQN", name="Duelling-DQN")
auto_save = 100
csv_file = "training.csv"

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.memory = deque(maxlen=capacity)
        self.device = device

    def insert(self, transition):
        transition = Transition(*[item.to('cpu') for item in transition])
        self.memory.append(transition)

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size), "Not enough samples in memory"
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch))
        return (
            torch.cat(batch.state).to(self.device),
            torch.cat(batch.action).to(self.device),
            torch.cat(batch.reward).to(self.device),
            torch.cat(batch.next_state).to(self.device),
            torch.cat(batch.done).to(self.device),
        )

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, nb_warmup=10000,
                 nb_actions=None, memory_capacity=10000, batch_size=32, learning_rate=0.001):
        self.memory = ReplayMemory(memory_capacity, device=device)
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device).eval()

        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.gamma = 0.99
        self.nb_actions = nb_actions
        self.batch_size = batch_size

    def select_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1), device=self.device)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values, dim=1, keepdim=True)

    def _update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes):
        for episode in range(1, epochs + 1):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) \
                    if not torch.is_tensor(state) else state.unsqueeze(0).to(self.device)

            done = False
            episode_return = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
                done_tensor = torch.tensor([[done]], dtype=torch.float32)

                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) \
                    if not torch.is_tensor(next_state) else next_state.unsqueeze(0)

                self.memory.insert((state.cpu(), action.cpu(), reward_tensor, next_state_tensor, done_tensor))

                if self.memory.can_sample(self.batch_size):
                    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
                    qsa = self.model(states).gather(1, actions)
                    with torch.no_grad():
                        max_next_qsa = self.target_model(next_states).max(1, keepdim=True)[0]
                        targets = rewards + (1 - dones) * self.gamma * max_next_qsa
                    loss = F.mse_loss(qsa, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state_tensor.to(self.device)
                episode_return += reward

            print(f"Episode {episode}, Return: {episode_return:.2f}, Epsilon: {self.epsilon:.3f}")
            wandb.log({
                "Episode": episode,
                "Return": episode_return,
                "Epsilon": self.epsilon,
                "Loss": loss.item() if 'loss' in locals() else None
            })

            if episode % auto_save == 0:
                save_dir = os.path.join(os.getcwd(), "models")
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f"model_ep{episode}.pt")
                torch.save(self.model.state_dict(), model_path)
                print(f"[AUTOSAVE] Model saved to {model_path}")
                wandb.save(model_path)

                if not os.path.exists(csv_file):
                    pd.DataFrame(columns=["episode", "reward", "epsilon"]).to_csv(csv_file, index=False)
                df = pd.read_csv(csv_file)
                new_row = pd.DataFrame([[episode, episode_return, self.epsilon]],
                                       columns=["episode", "reward", "epsilon"])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(csv_file, index=False)

            self._update_epsilon()

            if episode % 10 == 0:
                self.model.save_model()
                print("Model Saved")

            if episode % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print("Target model updated")
