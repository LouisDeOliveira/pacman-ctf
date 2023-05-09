import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, observation_size:int, action_size:int, hidden_size:int=64) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.observation_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x:torch.Tensor):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == "__main__":
    observation_size = 10
    observation = torch.randn(1, observation_size)
    print(observation)
    model = SimpleModel(observation_size, 4)
    model.eval()
    action = model(observation)
    print(action)