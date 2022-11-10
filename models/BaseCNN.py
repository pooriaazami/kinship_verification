import torch
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1, 1)) #(in_channels, out_channels, filter_size, padding=(1, 1)) --> padding (1, 1) is used to prevend shrinking the image size
        self.conv2 = nn.Conv2d(16, 64, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, 3, padding=(1, 1))