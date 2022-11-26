import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.KinFaceDataset import KinFaecDataset, ToTensor
from models.SiameseNet import PretrainedSiameseNet


TrainKINFaceWI = 'data\\TrainKinFaceWITriplets.csv'
TrainKINFaceWII = 'data\\TrainKinFaceWIITriplets.csv'

ValidationKINFaceWI = 'data\\ValidationKinFaceWITriplets.csv'
ValidationKINFaceWII = 'data\\ValidationKinFaceWIITriplets.csv'

TestKINFaceWI = 'data\\TestKinFaceWITriplets.csv'
TestKINFaceWII = 'data\\TestKinFaceWIITriplets.csv'

torch.manual_seed(100)

def create_loss_function(model, alpha, l2_alpha):
    def triplet_loss(anchor, pos, neg):
        count, _ = anchor.shape
        anchor_pos_diff = torch.pow(torch.norm(anchor - pos, dim=1), 2)
        anchor_neg_diff = torch.pow(torch.norm(anchor - neg, dim=1), 2)
        # print(anchor_neg_diff)
        loss_val = anchor_pos_diff - anchor_neg_diff + alpha
        loss_val = torch.fmax(loss_val, torch.zeros_like(loss_val))
        loss_val = torch.sum(loss_val)

        fc1_params = torch.cat([x.view(-1) for x in model.fc1.parameters()])
        fc2_params = torch.cat([x.view(-1) for x in model.fc2.parameters()])

        loss_val += l2_alpha * torch.norm(fc1_params, 2) + l2_alpha * torch.norm(fc2_params, 2)

        pos_count = torch.sum(anchor_pos_diff >= alpha)
        neg_count = torch.sum(anchor_neg_diff < alpha)
        accuracy = (pos_count + neg_count) / (count * 2)
        
        return loss_val, accuracy, anchor_pos_diff.mean(), anchor_neg_diff.mean()
    return triplet_loss

def training_step(train_dataset, model, optimizer, loss_fn, device):
    total_loss_train = .0
    anchor_pos_total_mean = .0
    anchor_neg_total_mean = .0
    train_losses = []
    # counter = 0
    for batch in tqdm(train_dataset):
        optimizer.zero_grad()

        anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor_embeddings = model(anchor)
        pos_embeddings = model(pos)
        neg_embeddings = model(neg)

        loss, accurary, anchor_pos_mean, anchor_neg_mean = loss_fn(anchor_embeddings, pos_embeddings, neg_embeddings)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss_train += loss_val
        anchor_pos_total_mean += anchor_pos_mean
        anchor_neg_total_mean += anchor_neg_mean

        train_losses.append(total_loss_train)
        

        total_loss_train /= len(train_dataset)
        anchor_pos_mean /= len(train_dataset)
        anchor_neg_mean /= len(train_dataset)

    return total_loss_train, accurary, anchor_pos_mean, anchor_neg_mean, train_losses

def validate_model(validation_dataset, model, optimizer, loss_fn, device):
    with torch.no_grad():
        total_loss_validation = .0
        anchor_pos_total_mean = .0
        anchor_neg_total_mean = .0
        validation_losses = []

        for batch in tqdm(validation_dataset):
            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchor_embeddings = model(anchor)
            pos_embeddings = model(pos)
            neg_embeddings = model(neg)

            loss, accuracy, anchor_pos_mean, anchor_neg_mean = loss_fn(anchor_embeddings, pos_embeddings, neg_embeddings)
            total_loss_validation += loss.item()
            anchor_pos_total_mean += anchor_pos_mean
            anchor_neg_total_mean += anchor_neg_mean
            validation_losses.append(total_loss_validation)

        total_loss_validation /= len(validation_dataset)
        anchor_pos_mean /= len(validation_dataset)
        anchor_neg_mean /= len(validation_dataset)

        return total_loss_validation, accuracy, anchor_pos_mean, anchor_neg_mean, validation_losses



def train_model(train_dataset, validation_dataset, model, loss_fn, optimizer, epochs=10, device='cpu'):
    fig, ax = plt.subplots(1, 1)
    train_losses = []
    validation_losses = []

    for epoch in range(1, epochs+1):
        total_loss_train, accuracy, anchor_pos_total_mean, anchor_neg_total_mean, train_losses_for_epoch = training_step(train_dataset, model, optimizer, loss_fn, device)
        print(f'#Epoch {epoch} loss: {total_loss_train} accuracy: {accuracy.item() * 100}, (a,p).mean: {anchor_pos_total_mean}, (a,n).mean(): {anchor_neg_total_mean}')

        total_loss_validation, accuracy, anchor_pos_total_mean, anchor_neg_toal_mean, validation_losses_for_epoch = validate_model(validation_dataset, model, optimizer, loss_fn, device)
        print(f'validation loss: {total_loss_validation} accuracy: {accuracy * 100}, (a,p).mean: {anchor_pos_total_mean}, (a,n).mean(): {anchor_neg_total_mean}')

        train_losses.extend(train_losses_for_epoch)
        validation_losses.extend(validation_losses_for_epoch)
            
        torch.save(model.state_dict(), f'model_{epoch}.pth')
        

    return train_losses, validation_losses


def split_dataset(dataset):
    validation_count = int(len(dataset) * validation_split_rate)
    train, validation = torch.utils.data.random_split(dataset, [len(dataset) - validation_count, validation_count])

    return train, validation

def load_dataset(data_portion=-1, val_portion=-1, train_batch_size=256, validation_batch_size=256, test_batch_size=256): #=-1, validation_split=.05, train_batch_size=256, validation_batch_size=256):
    # Train
    train_dataset_kinfacewi = KinFaecDataset(csv_path=TrainKINFaceWI, transform=transforms.Compose([
        ToTensor()
    ]))

    train_dataset_kinfacewii = KinFaecDataset(csv_path=TrainKINFaceWII, transform=transforms.Compose([
        ToTensor()
    ]))

    train_concatinated_data = torch.utils.data.ConcatDataset([train_dataset_kinfacewi, train_dataset_kinfacewii])

    # Validation
    Validation_dataset_kinfacewi = KinFaecDataset(csv_path=ValidationKINFaceWI, transform=transforms.Compose([
        ToTensor()
    ]))

    Validation_dataset_kinfacewii = KinFaecDataset(csv_path=ValidationKINFaceWII, transform=transforms.Compose([
        ToTensor()
    ]))

    validation_concatinated_data = torch.utils.data.ConcatDataset([Validation_dataset_kinfacewi, Validation_dataset_kinfacewii])

    # Test
    test_dataset_kinfacewi = KinFaecDataset(csv_path=TestKINFaceWI, transform=transforms.Compose([
        ToTensor()
    ]))

    test_dataset_kinfacewii = KinFaecDataset(csv_path=TestKINFaceWII, transform=transforms.Compose([
        ToTensor()
    ]))

    test_concatinated_data = torch.utils.data.ConcatDataset([test_dataset_kinfacewi, test_dataset_kinfacewii])

    if data_portion != -1:
        idx = torch.randperm(len(train_concatinated_data))[:data_portion]
        train_concatinated_data = torch.utils.data.Subset(train_concatinated_data, idx)

    if val_portion != -1:
        idx = torch.randperm(len(validation_concatinated_data))[:val_portion]
        validation_concatinated_data = torch.utils.data.Subset(validation_concatinated_data, idx)

    # train, validation = split_dataset(concatinated_data, validation_split)

    train_dataloader = DataLoader(train_concatinated_data, batch_size=train_batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
    validation_dataloader = DataLoader(validation_concatinated_data, batch_size=validation_batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
    test_dataloader = DataLoader(test_concatinated_data, batch_size=test_batch_size, shuffle=True, num_workers=4, prefetch_factor=2)

    # print(len(train_dataloader), len(validation_dataloader), len(test_dataloader))
    return train_dataloader, validation_dataloader, test_dataloader

def main():
    print('Initializing variables...')
    
    train_dataloader, validation_dataloader, test_dataloader = load_dataset(data_portion=2000, val_portion=100)
    model = PretrainedSiameseNet(device='cuda').to('cuda')
    criterion = create_loss_function(model, .1, 0.01)
    optimizer = optim.Adam(model.parameters())

    # hparams = {
    #     'alpha': 0.o1
    # }
    print('Done')
    # print(len(train), len(validation))
    train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, device='cuda', epochs=50)


if __name__ == '__main__':
    main()
