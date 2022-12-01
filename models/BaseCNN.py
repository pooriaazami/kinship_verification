import torch
import torch.nn as nn

import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, embedding_size=64):
        super().__init__()

        self.embedding_size = embedding_size

        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1)) #(in_channels, out_channels, filter_size, padding=(1, 1)) --> padding (1, 1) is used to prevend shrinking the image size
        self.conv2 = nn.Conv2d(16, 64, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.batch_norm = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # self.fc1 = nn.Linear(8192, self.embedding_size)
        self.embedding = nn.Linear(8192, self.embedding_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.batch_norm(x)

        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.embedding(x))
        
        return x