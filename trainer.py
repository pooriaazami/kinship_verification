import torch
import torch.optim as optim

from utils.TrainUtils import create_loss_function, train_model, load_dataset
from models.SiameseNet import PretrainedSiameseNet


def main():
    print('Initializing variables...')
    
    train_dataloader, validation_dataloader, test_dataloader = load_dataset(data_portion=2000, val_portion=100)
    model = PretrainedSiameseNet(device='cuda').to('cuda')
    criterion = create_loss_function(model, .1, 0.01)
    optimizer = optim.Adam(model.parameters(), le=0.001, weight_decat=0.01)

    print('Done')
    train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, device='cuda', epochs=50)


if __name__ == '__main__':
    main()
