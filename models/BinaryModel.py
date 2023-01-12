import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, base_model, embedding_size):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(2 * embedding_size, 1)

    def forward(self, first, second):
        first_embedding = self.base_model(first)
        second_embedding = self.base_model(second)
        joint_embedding = torch.cat([first_embedding, second_embedding], dim=1)
        return self.classifier(joint_embedding)

class BinaryClassifier(nn.Module):

    def __init__(self, embedding_size=64, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, latent_dim)
        # self.fc2 = nn.Linear(latent_dim , latent_dim)
        self.fc3 = nn.Linear(latent_dim , 1)
        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        return self.fc3(x)
        
