import torch
import torch.optim as optim
import torch.nn as nn

from utils.TrainUtils import create_loss_function, train_model, load_dataset, augmented_training_step, \
                     create_mixed_loss, validate_augmented_model, train_binary_classifier, validate_model
from models.SiameseNet import PretrainedSiameseNet, SiameseNet
from models.BinaryModel import BinaryClassifier

import matplotlib.pyplot as plt


def main():
    print('Initializing variables...')
    
    train_dataloader, validation_dataloader = load_dataset(dataset_code='mixed')#, data_portion=2000, val_portion=100)
    model = SiameseNet(device='cuda', use_attention=True, in_channels=3, embedding_size=64).to('cuda')
    criterion = create_loss_function(model, .2, 0.01)
    # criterion = create_mixed_loss(model, 1.)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print('Done')
    train_loss, val_loss, train_acc, val_acc = train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, device='cuda', epochs=1)
    # train_loss, val_loss, train_acc, val_acc = train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, training_step=augmented_training_step, validation_step=validate_augmented_model , device='cuda', epochs=500)
    print('Train Part I compleated')

    classifier = BinaryClassifier().to('cuda')
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.01)#, weight_decay=1e-4)
    classifier_criterion = nn.BCEWithLogitsLoss()
    classifier_train_acc, classifier_val_acc = train_binary_classifier(classifier, model, classifier_optimizer, classifier_criterion, train_dataloader, validation_dataloader, 1, 'cuda')

    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(train_loss, color='red', label='train loss')
    # axs[0].plot(val_loss, color='blue', label='val loss')
    # plt.plot(train_acc, color='red', label='train accuracy')
    # plt.plot(val_acc, color='blue', label='val accuracy')
    # axs[0].legend()
    # axs[1].legend()
    # plt.legend()
    # plt.show()

def test_models():
    embedding_model = SiameseNet(device='cuda', use_attention=True, in_channels=3, embedding_size=64).to('cuda')
    classifier_model = BinaryClassifier().to('cuda')

    embedding_model.load_state_dict(torch.load('.\\logs\\final model\\5\\embedding_model_5.pth'))
    classifier_model.load_state_dict(torch.load('.\\logs\\final model\\5\\classifier_model_2.pth'))

    train_dataloader, validation_dataloader = load_dataset(dataset_code='kfi')#, data_portion=20000, val_portion=100)

    accuracy = validate_model(embedding_model, classifier_model, train_dataloader, 'cuda')
    print(f'final accuracy: {accuracy * 100}')

    


if __name__ == '__main__':
    main()
    # test_models()
