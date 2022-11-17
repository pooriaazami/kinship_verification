import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import io, transform
import pandas as pd


class KinFaecDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.__triplets = pd.read_csv(csv_path)
        # print(self.__triplets.head())
        self.__transform = transform

    def __len__(self):
        return len(self.__triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        items = self.__triplets.iloc[idx]
        anchor_paths = list(items['parent'])
        posotive_paths = list(items['child'])
        negative_paths = list(items['negative_child'])

        anchor_images = io.imread_collection(anchor_paths)
        positive_images = io.imread_collection(posotive_paths)
        negative_images = io.imread_collection(negative_paths)

        if self.__transform:
            anchor_images = self.__transform(anchor_images)
            positive_images = self.__transform(positive_images)
            negative_images = self.__transform(negative_images)

        return {'anchor': anchor_images, 'pos': positive_images, 'neg': negative_images}

class ToTensor:
    def __call__(self, sample):
        anchor, pos, neg = sample['anchor'], sample['pos'], sample['neg']

        anchor = anchor.transpose((2, 0, 1))
        pos = pos.transpose((2, 0, 1))
        neg = neg.transpose((2, 0, 1))

        return {
            'anchor': torch.from_numpy(anchor),
            'pos': torch.from_numpy(pos),
            'neg': torch.from_numpy(neg)
        }

