import torch
import torch.nn as nn
import os 

class DuelingDQN(nn.Module):
    def __init__(self, nb_actions=4):
        super(DuelingDQN, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))  # changed kernel size to match common setups

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)

        self.flattened_size = self._get_flattened_size()

        self.action_value1 = nn.Linear(self.flattened_size, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_actions)

        self.state_value1 = nn.Linear(self.flattened_size, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 84, 84)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)

        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.state_value3(state_value)

        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)

        q_value = state_value + (action_value - action_value.mean(dim=1, keepdim=True))
        return q_value

    def save_model(self, filename='models/latest.pt'):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename='models/latest.pt'):
        if not os.path.exists(filename):
            print(" No saved model found, starting fresh.")
            return
        try:
            self.load_state_dict(torch.load(filename))
            print(f"Successfully loaded {filename}")
        except Exception as e:
            print(f"Failed to load model: {e}")
