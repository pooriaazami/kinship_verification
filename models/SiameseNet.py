import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseCNN import BaseCNN
from models.Attention import AttentionLayer, SpatialAttn
from models.VGGFace import VGGFace


class SiameseNet(nn.Module):
    def __init__(self, embedding_size=64, in_channels=3, device='cpu', use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.device = device

        if self.use_attention:
            self.attention_layer = SpatialAttn(in_channels)
            self.attention_mask = nn.Conv2d(3, 3, 3, padding=(1, 1))

        self.base_cnn = BaseCNN(embedding_size=embedding_size, in_channels=in_channels)
        # self.post_attention = nn.MultiheadAttention(embedding_size, 8)

        if self.use_attention:
            self.mask = torch.zeros((64, 64), dtype=torch.float32).to(self.device)
            self.mask[0:20, :] = 1
            self.mask[10:50, 20:40] = 1
            self.mask[50:, :] = 1

            self.mask = nn.Parameter(self.mask)
            self.mask.requires_grad = False

    def forward(self, x):
        if self.use_attention:
            learnable_attention = self.attention_layer(x)
            # print(x.shape)
            # print((x * self.mask).shape)
            mask_attention = x * self.mask
            # print(x.shape)
            # print(learnable_attention.shape)
            # print(mask_attention.shape)
            x = self.attention_mask(learnable_attention + mask_attention)
            x = F.relu(x)

        x = self.base_cnn(x)

        return x
    
    # def eval(self):
    #     super().eval()

    #     self.base_cnn.activate_bn_ema()

class PretrainedSiameseNet(nn.Module):
    def __init__(self, embedding_size=64, device='cpu', use_attention=False, freeze=False):
        super().__init__()
        self.device = device
        self.use_attention = use_attention

        if self.use_attention:
            self.attention_layer = SpatialAttn(3)
            self.attention_mask = nn.Conv2d(3, 3, 3, padding=(1, 1))
            # self.post_attention = nn.MultiheadAttention(8192, 4, dropout=.3)

        self.vggface = VGGFace(freeze=freeze)
        self.vggface.load_state_dict(torch.load('models\\pretrained\\VGGFace2.pt'))

        # self.fc1 = nn.Linear(8192, embedding_size)
        self.fc2 = nn.Linear(8192, embedding_size)

        self.dropout = nn.Dropout(.2)

        self.leaky_relu = nn.LeakyReLU()
        # self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        if self.use_attention:
            self.mask = torch.zeros((64, 64), dtype=torch.float32).to(self.device)
            self.mask[0:20, :] = 1
            self.mask[10:50, 20:40] = 1
            self.mask[50:, :] = 1


    def forward(self, x):
        if self.use_attention:
            learnable_attention = self.attention_layer(x)
            mask_attention = x * self.mask

            x = self.attention_mask(learnable_attention + mask_attention)

        x = self.vggface(x)

        x = torch.flatten(x, 1)
        # x = self.average_pool(x)
        # x = x.view(-1, 512)
        # x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # a, _ = self.post_attention(x, x, x)
        # x = a + x
        x = self.fc2(x)

        return x
