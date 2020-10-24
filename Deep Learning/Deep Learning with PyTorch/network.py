import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=5,
        )

        self.fc1 = nn.Linear(
            in_features=12 * 4 * 4,
            out_features=120,
        )
        self.fc2 = nn.Linear(
            in_features=120,
            out_features=60,
        )
        self.out = nn.Linear(
            in_features=60,
            out_features=10
        )

    def forward(self, t):
        # First Convolutional Hidden Layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # Second Convolutional Hidden Layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # First Linear Layer (needs reshaping!)
        t = t.reshape(-1, 12 * 4 * 4)
        # reshape the tensor, so it has 12 * 4 * 4 input features (axis=1) and corresponding number of outputs (axis=0)
        t = self.fc1(t)
        t = F.relu(t)

        # Second Linear Layer (does not need reshaping, it was reshaped during the transition between conv to linear)
        t = self.fc2(t)
        t = F.relu(t)

        # Output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1) # dim=1 as input because dim=0 is the output.
        # we do not use the softmax operation there, because our loss function cross_entropy uses it for us.

        return t