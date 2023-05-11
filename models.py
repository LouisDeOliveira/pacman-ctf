import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(
        self, observation_size: int, action_size: int, hidden_size: int = 64
    ) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.observation_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.action_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # add average pooling layer
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(84, 5)

    def forward(self, observation: torch.Tensor):
        x = self.relu(self.conv1(observation))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = torch.mean(x, dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    observation_size = 10
    observation = torch.randn(1, observation_size)
    print(observation)
    model = SimpleModel(observation_size, 4)
    model.eval()
    action = model(observation)
    print(action)
