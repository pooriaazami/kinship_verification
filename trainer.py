import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from utils.KinFaceDataset import KinFaecDataset, ToTensor
from models.SiameseNet import SiameseNet


KINFaceWI = 'data\\KinFaceWITriplets.csv'
KINFaceWII = 'data\\KinFaceWIITriplets.csv'


def create_loss_function(alpha):
    def triplet_loss(anchor, pos, neg):
        loss_val = torch.pow(torch.norm(anchor - pos, dim=1), 2) - \
            torch.pow(torch.norm(anchor - neg, dim=1), 2) + alpha
        loss_val = torch.fmax(loss_val, torch.zeros_like(loss_val))
        loss_val = torch.sum(loss_val)

        return loss_val

    return triplet_loss


def train(dataset, model, loss_fn, optimizer, epochs=10):
    losses = []

    for epoch in range(1, epochs+1):
        
        total_loss = .0
        for batch in dataset:
            optimizer.zero_grad()

            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']
            anchor_embeddings = model(anchor)
            pos_embeddings = model(pos)
            neg_embeddings = model(neg)

            loss = loss_fn(anchor_embeddings, pos_embeddings, neg_embeddings)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            losses.append(loss_val)

        if epoch % 10 == 0:
            print(f'#Epoch {epoch} loss: {loss_val}')
            # print(f'#Epoch {epochs} loss: {losses[-1]}')

    return losses


def main():
    print('Initializing variables...')
    dataset = KinFaecDataset(csv_path=KINFaceWI, transform=transforms.Compose([
        ToTensor()
    ]))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SiameseNet()
    criterion = create_loss_function(.01)
    optimizer = optim.Adam(model.parameters())

    # hparams = {
    #     'alpha': 0.o1
    # }
    print('Done')
    train(dataloader, model, criterion, optimizer)


if __name__ == '__main__':
    main()
