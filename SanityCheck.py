import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from utils.TrainUtils import create_loss_function, train_model, load_dataset
from models.SiameseNet import BaseCNN
from models.BinaryModel import Classifier

from sklearn.metrics import accuracy_score

from tqdm import tqdm


def generate_loss(loss_fn):
    def binary_loss_function(positive_samples, negative_samples):
        predictions = torch.cat([positive_samples, negative_samples], dim=0)
        true_labels = torch.cat([
            torch.ones_like(positive_samples),
            torch.zeros_like(negative_samples)
        ], dim=0)
        # print(predictions, true_labels)
        loss = loss_fn(predictions, true_labels)
        return loss

    return binary_loss_function
    

def calculate_accuracy(positive_samples, negative_samples):
    predictions = torch.cat([positive_samples, negative_samples], dim=0)
    predictions = torch.sigmoid(predictions) > 0.5
    true_labels = torch.cat([
        torch.ones_like(positive_samples),
        torch.zeros_like(negative_samples)
    ], dim=0)
    # print(predictions)
    # print(true_labels)
    return accuracy_score(predictions.cpu().numpy(), true_labels.cpu().numpy())



def train(train_dataloader, validation_dataloader, model, loss_fn, optimizer, device, epochs):
    total_accuracy = .0
    for epoch in range(1, epochs+1):
        total_loss = .0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_predictions = model(anchor, pos)
            neg_predictions = model(anchor, neg)

            loss = loss_fn(pos_predictions, neg_predictions)
            total_accuracy += calculate_accuracy(pos_predictions, neg_predictions)
            # print(calculate_accuracy(pos_predictions, neg_predictions))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()


        # with torch.no_grad():
        total_accuracy = .0
        for batch in tqdm(validation_dataloader):
            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_predictions = torch.sigmoid(model(anchor, pos))
            neg_predictions = torch.sigmoid(model(anchor, neg))

            loss = calculate_accuracy(pos_predictions, neg_predictions)
            total_accuracy += loss.item()


        print(f'#Epoch {epoch}, loss={total_loss / len(train_dataloader)}')
        print(f'Train accuracy = {total_accuracy / len(train_dataloader)}')
        print(f'Validation accuracy: {total_accuracy / len(validation_dataloader)}')

def main():
    print('Initializing variables...')
    
    train_dataloader, validation_dataloader, test_dataloader = load_dataset(dataset_code='kfii', data_portion=2000, val_portion=100)
    # model = PretrainedSiameseNet(device='cuda').to('cuda')
    base_model = BaseCNN()
    classifier = Classifier(base_model, 128).to('cuda')

    base_loss_fn = nn.BCEWithLogitsLoss()
    # base_val_fn = nn.BCELoss()

    criterion = generate_loss(base_loss_fn)
    # val_criterion = generate_loss(base_val_fn)
    optimizer = optim.Adam(classifier.parameters(), lr=1**-3)
    # print(list(classifier.parameters()))
    print('Done')
    train(train_dataloader, validation_dataloader, classifier, criterion, optimizer, device='cuda', epochs=50)


if __name__ == '__main__':
    main()
