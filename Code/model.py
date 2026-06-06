import torch
import torch.nn as nn


class DNA1DCNN(nn.Module):
    """
    Strict 5-layer 1D CNN according to the guideline:
    1) Conv1D Layer 1 + ReLU + MaxPool
    2) Conv1D Layer 2 + ReLU + MaxPool
    3) Conv1D Layer 3 + ReLU + MaxPool
    4) Dense Layer 1
    5) Dense Layer 2 with single neuron + Sigmoid
    """
    def __init__(self, kernel_size=5, seq_len=174, hidden_dim=256):
        super().__init__()

        # Conv1D Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Conv1D Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Conv1D Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Compute flattened size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 4, seq_len)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            flattened_dim = x.view(1, -1).shape[1]

        # Dense Layer 1
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Dense Layer 2 (output layer)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
