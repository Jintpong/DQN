import gymnasium as gym
import torch
import ale_py
import random
import numpy as np
import supersuit
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium.wrappers import AtariPreprocessing
import os
import wandb
import pandas as pd

# Initialize Weights & Biases
wandb.init(project="CW2-DQN", name="Duelling-DQN-3")
auto_save = 100  # Save model every 100 episodes
save_file = "training.csv"

# --- Define Dueling DQN Model ---
class DuelingDQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )

        self.fc_input_dim = self._get_conv_output_dim()

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(),
            nn.Linear(512, nb_actions)
        )

    def _get_conv_output_dim(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 4, 84, 84)
            conv_out = self.conv(sample_input)
            return int(np.prod(conv_out.size()[1:]))

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

# --- Deep Q-Learning function ---
def DuellingDQN(env, max_episodes=5000, replay_memory_size=1_000_000, update_frequency=4, batch_size=32,
                discount_factor=0.99, replay_start_size=5000, initial_exploration=1.0, final_exploration=0.01,
                device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    rb = ReplayBuffer(replay_memory_size, env.observation_space, env.action_space, device,
                      optimize_memory_usage=True, handle_timeout_termination=False)

    q_network = DuelingDQN(env.action_space.n).to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)

    epsilon = initial_exploration
    epsilon_decay = (initial_exploration - final_exploration) / max_episodes

    smoothed_rewards = []
    rewards = []
    frame_count = 0

    # Prepare CSV file
    if not os.path.exists(save_file):
        pd.DataFrame(columns=["episode", "reward", "epsilon"]).to_csv(save_file, index=False)

    progress_bar = tqdm(total=max_episodes)

    for episode in range(1, max_episodes + 1):
        obs, _ = env.reset()
        total_rewards = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = np.sign(reward)  # Reward clipping
            rb.add(obs, next_obs, np.array([action]), reward, done, info)

            obs = next_obs
            total_rewards += reward
            frame_count += 1

            if frame_count > replay_start_size and frame_count % update_frequency == 0:
                data = rb.sample(batch_size)
                with torch.no_grad():
                    next_q_values = q_network(data.next_observations).max(1)[0]
                    targets = data.rewards.flatten() + discount_factor * next_q_values * (1 - data.dones.flatten())
                q_pred = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.huber_loss(targets, q_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(final_exploration, epsilon - epsilon_decay)
        rewards.append(total_rewards)
        smoothed_rewards.append(np.mean(rewards[-100:]))

        progress_bar.set_description(f"Episode {episode}: Reward {total_rewards:.2f}, Epsilon {epsilon:.3f}")
        progress_bar.update(1)

        # Log to Weights & Biases
        wandb.log({
            "episode": episode,
            "episode_reward": total_rewards,
            "running_reward": smoothed_rewards[-1],
            "epsilon": epsilon
        })

        # Save model every 100 episodes
        if episode % auto_save == 0:
            save_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"model_ep{episode}.pt")
            torch.save(q_network.state_dict(), model_path)
            print(f"[AUTOSAVE] Model saved to {model_path}")

        # Save CSV progress
        df = pd.read_csv(save_file)
        new_row = pd.DataFrame([[episode, total_rewards, epsilon]], columns=["episode", "reward", "epsilon"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(save_file, index=False)

    env.close()

# --- Main ---
if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=1, render_mode=None)
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=True)
    env = supersuit.frame_stack_v1(env, 4)

    DuellingDQN(env)
