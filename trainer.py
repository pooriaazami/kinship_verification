import torch
import torch.optim as optim
import torch.nn as nn

from utils.TrainUtils import create_loss_function, train_model, load_dataset, augmented_training_step, \
                     create_mixed_loss, validate_augmented_model, train_binary_classifier, validate_model, \
                      generate_embeddings, load_splited_dataset, create_mixed_image_loss_function, \
                      train_mixed_image_network
from models.SiameseNet import PretrainedSiameseNet, SiameseNet, MobileNet, CombinedNetwork, MixedImageNetwork
from models.BinaryModel import BinaryClassifier

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    print('Initializing variables...')
    
    train_dataloader, validation_dataloader = load_dataset(dataset_code='kfi')#, data_portion=2000, val_portion=100)
    # embedding_dataloader, classification_dataloader, validation_dataloader = load_splited_dataset(dataset_code='kfi')#, data_portion=2000, val_portion=100)
    model = SiameseNet(device='cuda', use_attention=True, in_channels=3, embedding_size=128).to('cuda')
    # model = PretrainedSiameseNet(device='cuda', use_attention=True, embedding_size=128, freeze=True).to('cuda')
    # model = MobileNet(embedding_size=128, use_attention=False).to('cuda')
    # model = CombinedNetwork(embedding_size=64).to('cuda')
    # model = MixedImageNetwork(device='cuda', use_attention=True, in_channels=3, embedding_size=128).to('cuda')

    criterion = create_loss_function(model, .1, 0.01)
    # criterion = create_mixed_loss(model, 1.)
    # criterion = create_mixed_image_loss_function(model, .1, .01)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    print('Done')
    # model.load_state_dict(torch.load('.\\final_model_kinfacewii.pth'))
    train_loss, val_loss, train_acc, val_acc = train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, device='cuda', epochs=10)
    # train_loss, val_loss, train_acc, val_acc = train_model(train_dataloader, validation_dataloader, model, criterion, optimizer, training_step=augmented_training_step, validation_step=validate_augmented_model , device='cuda', epochs=500)
    # train_acc, val_acc = train_mixed_image_network(model, optimizer, criterion, train_dataloader, validation_dataloader, 50, 'cuda')
    # print('Train Part I compleated')

    # classifier = BinaryClassifier(embedding_size=64, latent_dim=128).to('cuda')
    # classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    # classifier_criterion = nn.BCEWithLogitsLoss()
    # classifier_train_acc, classifier_val_acc = train_binary_classifier(classifier, model, classifier_optimizer, classifier_criterion, train_dataloader, validation_dataloader, 20, 'cuda')

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
    # embedding_model = SiameseNet(device='cuda', use_attention=True, in_channels=3, embedding_size=64).to('cuda')
    # classifier_model = BinaryClassifier().to('cuda')
    embedding_model = MobileNet(embedding_size=128, use_attention=True, device='cuda').to('cuda')

    embedding_model.load_state_dict(torch.load('.\\embedding_model_4.pth'))
    # classifier_model.load_state_dict(torch.load('.\\classifier_model_1.pth'))

    train_dataloader, validation_dataloader = load_dataset(dataset_code='kfi')#, data_portion=20000, val_portion=100)

    accuracy = validate_model(embedding_model, None, validation_dataloader, 'cuda')
    print(f'final accuracy: {accuracy * 100}')
    # embeddings = generate_embeddings(embedding_model, train_dataloader, 'cuda')
    # embeddings = np.array(embeddings)
    # np.save('embeddings.npy', embeddings)
    # print(embeddings)
    # pd.DataFrame.from_dict(embeddings['anchor']).to_csv('train_anchor_log.csv')
    # print('anchots saved')
    # pd.DataFrame.from_dict(embeddings['pos']).to_csv('train_pos_log.csv')
    # print('poses saved')
    # pd.DataFrame.from_dict(embeddings['neg']).to_csv('train_neg_log.csv')
    # print('negs saved')

    

    


if __name__ == '__main__':
    main()
    # test_models()
