import os
import ale_py
import pandas as pd 
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import AtariPreprocessing
from supersuit import frame_stack_v1
import wandb


wandb.init(project="CW2-DQN", name="Duelling-DQN-2")
# --- Hyperparameters ---

auto_save = 100  
save_file = "training.csv"
seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_decay = (epsilon_max - epsilon_min) / 1_000_000 
batch_size = 32
learning_rate = 0.001
max_steps_per_episode = 10000
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000

# --- Environment Setup ---
env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=1, render_mode=None)

env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
env = frame_stack_v1(env, 4)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define the DQN model ---
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )

        self.fc_input_dim = self._get_conv_output(input_shape)

        self.fc_value = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(),
            nn.Linear(512, 1)  
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(),
            nn.Linear(512, num_actions)  
        )

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x / 255.0  
        features = self.conv(x)
        features = features.view(features.size(0), -1)

        value = self.fc_value(features)
        advantage = self.fc_advantage(features)

        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

num_actions = env.action_space.n
policy_net = DuelingDQN((4, 84, 84), num_actions).to(device)
target_net = DuelingDQN((4, 84, 84), num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.HuberLoss()

# --- Replay buffer ---
replay_memory = deque(maxlen=max_memory_length)

# --- Training Loop ---
frame_count = 0
episode_count = 0
episode_reward_history = []
running_reward = 0
max_episodes = 2000 

while True:
    obs, _ = env.reset()
    state = np.array(obs, copy=False)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        if frame_count < 50_000 or random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        next_obs, reward, done, truncated, _ = env.step(action)
        next_state = np.array(next_obs, copy=False)

        replay_memory.append((state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward

        # Update epsilon
        if epsilon > epsilon_min:
            epsilon -= epsilon_decay
        else:
            epsilon = epsilon_min

        # Sample and train
        if frame_count % update_after_actions == 0 and len(replay_memory) > batch_size:
            minibatch = random.sample(replay_memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(device)
            dones = torch.FloatTensor(dones).to(device)

            q_values = policy_net(states).gather(1, actions).squeeze()
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (1 - dones)

            loss = loss_fn(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the target network
        if frame_count % update_target_network == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"[Update] Target network updated at frame {frame_count}")

        if done or truncated:
            break

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        episode_reward_history.pop(0)

    running_reward = np.mean(episode_reward_history)
    episode_count += 1

    print(f"Episode {episode_count}: Return: {episode_reward:.2f}, Epsilon: {epsilon:.3f}, Running Reward: {running_reward:.2f}")

    wandb.log({
    "Episode": episode_count,
    "Episode_reward": episode_reward,
    "Running_reward": running_reward,
    "epsilon": epsilon
    })

    if (episode_count) % auto_save == 0:
        save_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"model_ep{episode_count}.pt")
        torch.save(policy_net.state_dict(), model_path)
        print(f"[AUTOSAVE] Model saved to {model_path}")

    # --- Save CSV ---
    if not os.path.exists(save_file):
        pd.DataFrame(columns=["episode", "reward", "epsilon"]).to_csv(save_file, index=False)

    df = pd.read_csv(save_file)
    new_row = pd.DataFrame([[episode_count, episode_reward, epsilon]], columns=["episode", "reward", "epsilon"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(save_file, index=False)

    if running_reward > 40:
        print(f"Solved at episode {episode_count}!")
        break

    if max_episodes > 0 and episode_count >= max_episodes:
        print(f"Stopped at episode {episode_count}")
        break
