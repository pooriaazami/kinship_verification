import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from utils.KinFaceDataset import KinFaecDataset, ToTensor
from models.SiameseNet import SiameseNet


KINFaceWI = 'data\\KinFaceWITriplets.csv'
KINFaceWII = 'data\\KinFaceWIITriplets.csv'


dataset = KinFaecDataset(csv_path=KINFaceWI, transform=transforms.Compose([
    ToTensor()
]))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# print(dataset[0])

model = SiameseNet()

data = next(iter(dataloader))
anchor, pos, neg = data['anchor'], data['pos'], data['neg']
print(anchor.shape)