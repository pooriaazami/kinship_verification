import torch
import torch.nn as nn

import torch.nn.functional as F

from models.Attention import AttentionLayer, SpatialAttn

class BaseCNN(nn.Module):
    def __init__(self, embedding_size=64, in_channels=3):
        super().__init__()

        self.embedding_size = embedding_size

        self.conv1_1 = nn.Conv2d(in_channels, 32, 3, padding=(1, 1)) #(in_channels, out_channels, filter_size, padding=(1, 1)) --> padding (1, 1) is used to prevend shrinking the image size
        # self.conv1_2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        # self.res_conv_1 = nn.Conv2d(in_channels, 32, 3, padding=(1, 1))
        self.batch_norm_1 = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        # self.conv2_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        # self.res_conv_2 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        # self.conv3_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))
        # self.res_conv_3 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.batch_norm_3 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        # self.conv4_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))
        # self.res_conv_4 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.batch_norm_4 = nn.BatchNorm2d(256)

        self.small_pool = nn.MaxPool2d(2, 2)
        self.large_pool = nn.MaxPool2d(4, 4)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fc1 = nn.Linear(8192, self.embedding_size)
        # self.fc1 = nn.Linear(4096, self.embedding_size)
        self.fc2 = nn.Linear(256, self.embedding_size)
        self.dropout = nn.Dropout(.5)
        # self.post_attention = nn.MultiheadAttention(256, 4, dropout=.3)
        # self.middle_attention = SpatialAttn(32)
        # self.leaky_relu = torch.nn.LeakyReLU()


    def forward(self, x):
        # img = x
        x = F.relu(self.conv1_1(x))
        # y = F.relu(self.res_conv_1(img))
        # x = F.relu(self.conv1_2(x))
        # x = x + y
        x = self.small_pool(x)
        # a    = self.middle_attention(x)
        # x = a + x
        # x = self.batch_norm_1(x)
       
        x = F.relu(self.conv2_1(x))
        # x = F.relu(self.conv2_2(x))
        # x = x + y
        x = self.small_pool(x)
        # x = self.batch_norm_2(x)
        
        x = F.relu(self.conv3_1(x))
        # x = F.relu(self.conv3_2(x))
        # x = x + y
        x = self.small_pool(x)
        x = self.batch_norm_3(x)

        
        # y = self.pool(y)
        # y = F.relu(self.res_conv_2(y))
        # y = self.pool(y)
        # y = F.relu(self.res_conv_3(y))
        # y = self.pool(y)

        # x = x + y
        x = F.relu(self.conv4_1(x))
        # x = F.relu(self.conv4_2(x))
        # x = x + y
        x = self.small_pool(x)
        x = self.batch_norm_4(x)
        
        # x = self.attention(x)
        
        

        # x = torch.flatten(x, 1)
        x = self.average_pool(x)
        x = x.view(-1, 256)
        # print(x.shape)
        # x = self.dropout(x)
        # a, _ = self.post_attention(x, x, x)
        # print(a.shape)
        # x = a + x
        x = self.fc2(x)
        
        return x