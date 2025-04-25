import gymnasium as gym
import numpy as np
import ale_py
import torch
from PIL import Image 
from breakout import *
from model import *
from agent import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

env = Breakout()

model = DuelingDQN(nb_actions = 4)
model.load_model()

agent = Agent(
    model=model,
    device=device,
    epsilon=1.0,
    nb_warmup=5000,
    nb_actions=4,
    learning_rate=0.0001,
    memory_capacity=1000000,
    batch_size=64
)

agent.train(env, episodes=2500) 