import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseCNN import BaseCNN
from models.Attention import AttentionLayer
from models.VGGFace import VGGFace


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_layer = AttentionLayer(3)
        self.base_cnn = BaseCNN()

    def forward(self, x):
        x = self.attention_layer(x)
        x = self.base_cnn(x)

        return x

class PretrainedSiameseNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.attention_layer = AttentionLayer(3)
        self.vggface = VGGFace(trainable=False)

        self.fc1 = nn.Linear(512 * 2 * 2, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        x = self.attention_layer(x)
        x = self.vggface(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
