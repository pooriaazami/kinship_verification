import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform
import pandas as pd


class KinFaceDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.__triplets = pd.read_csv(csv_path)
        self.__transform = transform

    def __len__(self):
        return len(self.__triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        items = self.__triplets.iloc[idx]
        anchor_paths = items['parent']
        posotive_paths = items['child']
        negative_paths = items['negative_child']

        anchor_images = io.imread(anchor_paths)
        positive_images = io.imread(posotive_paths)
        negative_images = io.imread(negative_paths)

        sample = {'anchor': anchor_images,
                  'pos': positive_images,
                  'neg': negative_images}

        if self.__transform:
            # anchor_images = self.__transform(anchor_images)
            # positive_images = self.__transform(positive_images)
            # negative_images = self.__transform(negative_images)
            sample = self.__transform(sample)

        return sample

class ToTensor:
    def __call__(self, sample):
        anchor, pos, neg = sample['anchor'], sample['pos'], sample['neg']

        anchor = anchor.transpose((2, 0, 1)).astype(np.float32)
        pos = pos.transpose((2, 0, 1)).astype(np.float32)
        neg = neg.transpose((2, 0, 1)).astype(np.float32)

        return {
            'anchor': torch.from_numpy(anchor),
            'pos': torch.from_numpy(pos),
            'neg': torch.from_numpy(neg)
        }

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
        # self.mean = torch.Tensor(mean, device=device)
        # self.std = torch.Tensor(std, device=device)
        self.transform = transforms.Normalize(mean, std)

    def __call__(self, sample):
        anchor, pos, neg = sample['anchor'], sample['pos'], sample['neg']

        anchor = self.transform(anchor) #(anchor - self.mean) / self.std
        pos = self.transform(pos) #(pos - self.mean) / self.std
        neg = self.transform(neg) #(neg - self.mean) / self.std

        return {
            'anchor': anchor,
            'pos': pos,
            'neg': neg
        }
