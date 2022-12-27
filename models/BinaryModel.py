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

class BinaryClassifier(nn.Module):

    def __init__(self, embedding_size=64):
        super().__init__()

        self.fc1 = nn.Linear(embedding_size * 2, 64)
        # self.fc2 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(64 , 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = torch.cat((e1, e1), dim=1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc2(x)
        
