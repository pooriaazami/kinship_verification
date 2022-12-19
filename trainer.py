import torch
import torch.optim as optim

from utils.TrainUtils import create_loss_function, train_model, load_dataset, augmented_training_step, create_mixed_loss, validate_augmented_model
from models.SiameseNet import PretrainedSiameseNet, SiameseNet


def main():
    print('Initializing variables...')
    
    train_dataloader, validation_dataloader = load_dataset(dataset_code='kfii')#, data_portion=100, val_portion=50)
    model = SiameseNet(device='cuda', use_attention=True, in_channels=4).to('cuda')
    # criterion = create_loss_function(model, 1., 0.01)
    criterion = create_mixed_loss(model, 1.)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.05)

    print('Done')
    # train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, device='cuda', epochs=500)
    train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, training_step=augmented_training_step, validation_step=validate_augmented_model , device='cuda', epochs=500)


if __name__ == '__main__':
    main()
