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

        self.__cache = {}

    def __len__(self):
        return len(self.__triplets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        items = self.__triplets.iloc[idx]
        if items['parent'] not in self.__cache.keys():
            self.__cache[items['parent']] = io.imread(items['parent'])
        
        if items['child'] not in self.__cache.keys():
            self.__cache[items['child']] = io.imread(items['child'])

        if items['negative_child'] not in self.__cache.keys():
            self.__cache[items['negative_child']] = io.imread(items['negative_child'])

        anchor_images = self.__cache[items['parent']]
        positive_images = self.__cache[items['child']]
        negative_images = self.__cache[items['negative_child']]

        sample = {'anchor': anchor_images,
                  'pos': positive_images,
                  'neg': negative_images}

        if self.__transform:
            # anchor_images = self.__transform(anchor_images)
            # positive_images = self.__transform(positive_images)
            # negative_images = self.__transform(negative_images)
            sample = self.__transform(sample)

        return sample

# class KinFaceWII(Dataset):
#     def __init__(self, path,transform=None):
#         self.__base_path = path

#     def __len__(self):
#         return 500 * 4

#     def __generate_path(self, idx):
#         if idx < 500:
#             path = f'{self.__base_path}\\father-dau\\fd_{str(idx).rjust(3, "0")}_{idx%2}.jpg'
#             label = 'father-dau'
#         elif 500 <= idx < 1000:
#             idx -= 500
#             path = f'{self.__base_path}\\father-son\\fs_{str(idx).rjust(3, "0")}_{idx%2}.jpg'
#             label = 'father-son'
#         elif 1000 <= idx < 1500:
#             idx -= 1000
#             path = f'{self.__base_path}\\mother-dau\\md_{str(idx).rjust(3, "0")}_{idx%2}.jpg'
#             label = 'mother-dau'
#         elif 1500 <= idx < 2000:
#             idx -= 1500
#             path = f'{self.__base_path}\\mother-son\\ms_{str(idx).rjust(3, "0")}_{idx%2}.jpg'
#             label = 'mother-son'
#         else:
#             path = None

#         return path, label
    
#     def __getitem__(self, idx):
#         path, label = self.__generate_path(idx)
#         img = io.imread(path)

#         if self.__transform:
#             sample = self.__transform(sample)


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


class Augment:
    def __init__(self):
        self.transform = torch.nn.Sequential(
             transforms.RandomRotation(30),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip()
        )

    def __call__(self, sample):
        anchor, pos, neg = sample['anchor'], sample['pos'], sample['neg']

        anchor = self.transform(anchor)
        pos = self.transform(pos)
        neg = self.transform(neg)

        return {
            'anchor': anchor,
            'pos': pos,
            'neg': neg
        }

