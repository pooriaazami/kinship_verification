import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseCNN import BaseCNN
from models.Attention import AttentionLayer
from models.VGGFace import VGGFace


class SiameseNet(nn.Module):
    def __init__(self, embedding_size=64, in_channels=3, device='cpu', use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.device = device

        if self.use_attention:
            self.attention_layer = AttentionLayer(in_channels)

        self.base_cnn = BaseCNN(embedding_size=embedding_size, in_channels=in_channels)

        if self.use_attention:
            self.mask = torch.zeros((64, 64), requires_grad=False, dtype=torch.float32).to(self.device)
            self.mask[0:20, :] = 1
            self.mask[10:50, 20:40] = 1
            self.mask[50:, :] = 1

    def forward(self, x):
        if self.use_attention:
            learnable_attention = self.attention_layer(x)
            mask_attention = x * self.mask

            x = learnable_attention + mask_attention

        x = self.base_cnn(x)

        return x

class PretrainedSiameseNet(nn.Module):
    def __init__(self, embedding_size=64, device='cpu', use_attention=False, freeze=False):
        super().__init__()
        self.device = device
        self.use_attention = use_attention

        if self.use_attention:
            self.attention_layer = AttentionLayer(3)

        self.vggface = VGGFace(trainable=False, freeze=freeze)
        self.vggface.load_state_dict(torch.load('models\\pretrained\\VGGFace2.pt'))

        self.fc1 = nn.Linear(512 * 2 * 2, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(.5)

        self.leaky_relu = nn.LeakyReLU()

        if self.use_attention:
            self.mask = torch.zeros((64, 64), requires_grad=False, dtype=torch.float32).to(self.device)
            self.mask[10:20, 5:60] = 1
            self.mask[10:50, 27:37] = 1
            self.mask[50:60, 5:60] = 1



    def forward(self, x):
        if self.use_attention:
            learnable_attention = self.attention_layer(x)
            mask_attention = x * self.mask

            x = learnable_attention + mask_attention

        x = self.vggface(x)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x
