import torch
import torch.nn as nn

from BaseCNN import BaseCNN
from Attention import AttentionLayer


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_layer = AttentionLayer(3)
        self.base_cnn = BaseCNN()

    def forward(self, x):
        x = self.attention_layer(x)
        x = self.base_cnn(x)

        return x
