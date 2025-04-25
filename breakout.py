import gymnasium as gym
import cv2 
import numpy as np
import ale_py
import torch
from PIL import Image 
from torchvision import transforms



class Breakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array'):
        env = gym.make("ALE/Breakout-v5", render_mode=render_mode, frameskip=1)
        super().__init__(env)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info

    def _preprocess(self, obs):
        return self.transform(obs)