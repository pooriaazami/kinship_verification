import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseCNN import BaseCNN
from models.Attention import AttentionLayer, SpatialAttn
from models.VGGFace import VGGFace
from models.VGG import VGGModel


class SiameseNet(nn.Module):
    def __init__(self, embedding_size=64, in_channels=3, device='cpu', use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.device = device

        if self.use_attention:
            self.attention_layer = AttentionLayer(in_channels)
            self.attention_mask = nn.Conv2d(in_channels, in_channels, 3, padding=(1, 1))

        self.base_cnn = BaseCNN(embedding_size=embedding_size, in_channels=in_channels)
        # self.post_attention = nn.MultiheadAttention(embedding_size, 8)

        if self.use_attention:
            self.mask = torch.zeros((64, 64), dtype=torch.float32).to(self.device)
            self.mask[8:24,8:56] = 1
            self.mask[24:32,24:40] = 1
            self.mask[32:64,16:48] = 1

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
            self.attention_layer = AttentionLayer(3)

        self.vggface = VGGModel(False)
        self.vggface.load_state_dict(torch.load('models\\pretrained\\pretrained_vgg16.pth'))
        # print(self.vggface)
        if freeze:
            self.vggface.requires_grad = False

        self.fc1 = nn.Linear(4096, embedding_size)
        # self.fc2 = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(.2)

        # self.leaky_relu = nn.LeakyReLU()
        # if self.use_attention:
        #     self.mask = torch.zeros((64, 64), dtype=torch.float32).to(self.device)
        #     self.mask[8:24,8:56] = 1
        #     self.mask[24:32,24:40] = 1
        #     self.mask[32:64,16:48] = 1


    def forward(self, x):
        if self.use_attention:
            x = self.attention_layer(x)
            # mask_attention = x * self.mask

            # x = self.attention_mask(learnable_attention + mask_attention)

        x = self.vggface(x)
        x = F.relu(x)
        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = self.average_pool(x)
        # x = x.view(-1, 512)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # a, _ = self.post_attention(x, x, x)
        # x = a + x
        x = self.fc1(x)

        return x

    def unfreeze(self):
        self.vggface.requires_grad = True
        # self.vggface.unfreeze()


class MobileNet(nn.Module):
    def __init__(self, embedding_size=64, use_attention=False, device='cpu'):
        super().__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT')
        self.use_attention = use_attention
        self.device = device

        if self.use_attention:
            self.attention_layer = SpatialAttn(3)
            self.attention_mask = nn.Conv2d(3, 3, 3, padding=(1, 1))

        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(1000, embedding_size)

        self.base_model.requires_grad = False
        
        if self.use_attention:
            self.post_attention = nn.MultiheadAttention(1000, 4, dropout=.3)
            self.mask = torch.zeros((64, 64), dtype=torch.float32).to(self.device)
            self.mask[8:24,8:56] = 1
            self.mask[24:32,24:40] = 1
            self.mask[32:64,16:48] = 1

            self.mask = nn.Parameter(self.mask)
            self.mask.requires_grad = False

    def forward(self, x):
        # if self.use_attention:
        #     learnable_attention = self.attention_layer(x)
        #     # print(x.shape)
        #     # print((x * self.mask).shape)
        #     mask_attention = x * self.mask
        #     # print(x.shape)
        #     # print(learnable_attention.shape)
        #     # print(mask_attention.shape)
        #     x = self.attention_mask(learnable_attention + mask_attention)
        #     x = F.relu(x)

        x = self.base_model(x)
        # print(x.sh)
        x = torch.flatten(x, 1)
        # print(x.shape)
        if self.use_attention:
            a, _ = self.post_attention(x, x, x)
            x = a + x
        # x = self.dropout(x)
        x = self.fc1(x)
        return x


class CombinedNetwork(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.mobilenet  = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT')
        self.vggface = VGGFace(freeze=True)

        self.vggface.load_state_dict(torch.load('models\\pretrained\\VGGFace2.pt'))
        self.mobilenet.requires_grad = False
        self.vggface.requires_grad = False

        self.fc1 = nn.Linear(1000 + 8192, embedding_size)
        # self.post_attention = nn.MultiheadAttention(embedding_size, 4, dropout=.3)

    def forward(self, x):
        first_embedding_term = self.mobilenet(x)
        second_embedding_term = self.vggface(x)
        second_embedding_term = torch.flatten(second_embedding_term, 1)

        # print(first_embedding_term.shape, second_embedding_term.shape)
        embedding = torch.hstack([first_embedding_term, second_embedding_term])
        # print(embedding.shape)
        x = self.fc1(embedding)
        # x, _ = self.post_attention(embedding, embedding, embedding)
        # # print(x.shape, a.shape)
        # # x = self.fc1(embedding + x)
        # # x = x + a
        return x


class MixedImageNetwork(nn.Module):
    def __init__(self, embedding_size=64, in_channels=3, device='cpu', use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.device = device

        if self.use_attention:
            self.attention_layer = SpatialAttn(in_channels)
            self.attention_mask = nn.Conv2d(in_channels, in_channels, 3, padding=(1, 1))

        self.base_cnn = BaseCNN(embedding_size=embedding_size, in_channels=in_channels)
        self.fc1 = nn.Linear(embedding_size, 1)
        # self.post_attention = nn.MultiheadAttention(embedding_size, 8)

        if self.use_attention:
            self.mask = torch.zeros((64, 64), dtype=torch.float32).to(self.device)
            self.mask[8:24,8:56] = 1
            self.mask[24:32,24:40] = 1
            self.mask[32:64,16:48] = 1

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
        x = self.fc1(x)

        return x    