import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, base_model, embedding_size):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(2 * embedding_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, first, second):
        # first, second = x[0, :], x[1, :]
        first_embedding = self.base_model(first)
        second_embedding = self.base_model(second)
        # print(first_embedding.shape)
        joint_embedding = torch.cat([first_embedding, second_embedding], dim=1)
        # print(joint_embedding.shape)
        return self.classifier(joint_embedding)
        
