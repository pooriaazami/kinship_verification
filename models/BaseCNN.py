import torch
import torch.nn as nn

import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, embedding_size=64, in_channels=3):
        super().__init__()

        self.embedding_size = embedding_size

        self.conv1_1 = nn.Conv2d(in_channels, 16, 3, padding=(1, 1)) #(in_channels, out_channels, filter_size, padding=(1, 1)) --> padding (1, 1) is used to prevend shrinking the image size
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=(1, 1))
        self.batch_norm_1 = nn.BatchNorm2d(16)

        self.conv2_1 = nn.Conv2d(16, 64, 3, padding=(1, 1))
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        self.batch_norm_3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # self.fc1 = nn.Linear(8192, self.embedding_size)
        self.fc1 = nn.Linear(8192, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = self.batch_norm_1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = self.batch_norm_2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)
        x = self.batch_norm_3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x