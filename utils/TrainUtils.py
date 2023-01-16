import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import logging
import time

import random

# import pandas as pd

from utils.KinFaceDataset import KinFaceDataset, ToTensor, Normalize, Augment, Resize

TrainKINFaceWI = 'data\\KinFaceWITrainFolds.csv'
TrainKINFaceWII = 'data\\KinFaceWIITrainFolds.csv'

ValidationKINFaceWI = 'data\\KinFaceWITestFolds.csv'
ValidationKINFaceWII = 'data\\KinFaceWIITestFolds.csv'

# logging.basicConfig(filename=f'{int(time.time() * 1000)}.log', level=logging.DEBUG)

def euclidean_distances(vec1, vec2):
    vec1 = vec1 / torch.norm(vec1)
    vec2 = vec2 / torch.norm(vec2)

    distance = torch.pow(torch.norm(vec1 - vec2, dim=1), 2)
    return distance

def create_loss_function(model, alpha, l2_alpha):
    sim = torch.nn.CosineSimilarity()
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    # loss_alpha = .2
    dis = euclidean_distances
    def triplet_loss(anchor, pos, neg):
        count, _ = anchor.shape
        # print(count)
        anchor_pos_diff = 1 - sim(anchor, pos)
        anchor_neg_diff = 1 - sim(anchor, neg)
        # anchor_pos_diff = dis(anchor, pos)
        # anchor_neg_diff = dis(anchor, neg)

        # pos_class_mask = anchor_pos_diff - alpha # <= alpha
        # neg_class_mask = anchor_neg_diff - alpha # > alpha
        # pos_class_loss = loss_func(pos_class_mask, torch.ones_like(pos_class_mask))
        # neg_class_loss = loss_func(neg_class_mask, torch.zeros_like(neg_class_mask))
        
        # print(pos_class_loss.item())
        loss_val = anchor_pos_diff - anchor_neg_diff + alpha# + pos_class_loss + neg_class_loss
        loss_val = torch.fmax(loss_val, torch.zeros_like(loss_val))
        loss_val = torch.mean(loss_val) 

        with torch.no_grad():
            threshold = anchor_pos_diff.mean() + anchor_pos_diff.std() 
            pos_mask = anchor_pos_diff <= threshold
            neg_mask = anchor_neg_diff > threshold

            pos_count = torch.sum(pos_mask)
            neg_count = torch.sum(neg_mask)
            # print(pos_count, neg_count)
            accuracy = (pos_count + neg_count) / (count * 2)
            
        return loss_val, accuracy, anchor_pos_diff.mean(), anchor_neg_diff.mean(), anchor_pos_diff.std(), anchor_neg_diff.std()
    return triplet_loss

def create_mixed_image_loss_function(model, alpha, l2_alpha):
    # sim = torch.nn.CosineSimilarity()
    # dis = euclidean_distances
    loss_func = nn.BCEWithLogitsLoss()
    def triplet_loss(anchor_pos, anchor_neg):
        count, _ = anchor_pos.shape

        # anchor_pos_diff = 1 - sim(anchor, pos)
        # anchor_neg_diff = 1 - sim(anchor, neg)
        # anchor_pos_diff = dis(anchor, pos)
        # anchor_neg_diff = dis(anchor, neg)

        # anchor_pos = torch.sum(anchor_pos, dim=1)
        # anchor_neg = torch.sum(anchor_neg, dim=1)

        # loss_val = anchor_pos - anchor_neg + alpha
        # loss_val = torch.fmax(loss_val, torch.zeros_like(loss_val))
        # loss_val = torch.sum(loss_val)
        loss_term_1 = loss_func(anchor_pos, torch.ones_like(anchor_pos))
        loss_term_2 = loss_func(anchor_neg, torch.zeros_like(anchor_neg))
        loss_val = loss_term_1 + loss_term_2

        with torch.no_grad():
            # threshold = anchor_pos.mean() + anchor_pos.std() 
            pos_count = torch.sum(anchor_pos >= 0)
            neg_count = torch.sum(anchor_neg < 0)

            accuracy = (pos_count + neg_count) / (count * 2)
        
        return loss_val, accuracy, anchor_pos.mean(), anchor_neg.mean(), anchor_pos.std(), anchor_neg.std()
    return triplet_loss

def create_mixed_loss(model, alpha):
    sim = torch.nn.CosineSimilarity()
    # dis = euclidean_distances
    def triplet_loss(true_anchor, true_pos, true_neg, false_anchor, false_pos):
        count, _ = true_anchor.shape

        tanchor_tpos_diff = 1 - sim(true_anchor, true_pos)
        tanchor_tneg_diff = 1 - sim(true_anchor, true_neg)
        fanchor_fpos_diff = 1 - sim(false_anchor, false_pos)
        # tanchor_tpos_diff = dis(true_anchor, true_pos)
        # tanchor_tneg_diff = dis(true_anchor, true_neg)
        # fanchor_fpos_diff = dis(false_anchor, false_pos)

        first_term = tanchor_tpos_diff - tanchor_tneg_diff + alpha
        first_term = torch.fmax(first_term, torch.zeros_like(first_term))
        first_term = torch.sum(first_term)

        second_term = tanchor_tpos_diff - fanchor_fpos_diff + alpha
        second_term = torch.fmax(second_term, torch.zeros_like(second_term))
        second_term = torch.sum(second_term)

        loss_val = first_term + second_term

        with torch.no_grad():
            threshold = tanchor_tpos_diff.mean()
            tpos_count = torch.sum(tanchor_tpos_diff <= threshold)
            tneg_count = torch.sum(tanchor_tneg_diff > threshold)
            fpos_count = torch.sum(fanchor_fpos_diff > threshold)

            accuracy = (tpos_count + tneg_count) / (count * 2)

        ap_mean = tanchor_tpos_diff.mean()
        ap_std =  tanchor_tpos_diff.std()
        np_mean = (fanchor_fpos_diff + tanchor_tneg_diff).mean()
        np_std = (fanchor_fpos_diff + tanchor_tneg_diff).std()

        return loss_val, accuracy, ap_mean, np_mean, ap_std, np_std
    return triplet_loss

def training_step(train_dataset, model, optimizer, loss_fn, device):
    model.train()
    total_loss_train = .0
    total_loss_validation = .0
    anchor_pos_total_mean = .0
    anchor_neg_total_mean = .0
    anchor_pos_total_std = .0
    anchor_neg_total_std = .0
    total_accuracy = .0
    train_losses = []
    # counter = 0
    for batch in tqdm(train_dataset):
        optimizer.zero_grad()

        anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']
        # print(anchor.shape)

        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor_embeddings = model(anchor)
        pos_embeddings = model(pos)
        neg_embeddings = model(neg)

        loss, accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std = loss_fn(anchor_embeddings, pos_embeddings, neg_embeddings)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss_train += loss_val
        anchor_pos_total_mean += anchor_pos_mean
        anchor_neg_total_mean += anchor_neg_mean
        total_accuracy += accuracy

        train_losses.append(total_loss_train)
        

    total_loss_train /= len(train_dataset)
    anchor_pos_mean /= len(train_dataset)
    anchor_neg_mean /= len(train_dataset)
    anchor_pos_std /= len(train_dataset)
    anchor_neg_std /= len(train_dataset)
    total_accuracy /= len(train_dataset)

    return total_loss_train, total_accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std, train_losses

def augmented_training_step(train_dataset, model, optimizer, loss_fn, device):
    model.train()
    total_loss_train = .0
    total_loss_validation = .0
    anchor_pos_total_mean = .0
    anchor_neg_total_mean = .0
    anchor_pos_total_std = .0
    anchor_neg_total_std = .0
    total_accuracy = .0
    train_losses = []
    # counter = 0
    for batch in tqdm(train_dataset):
        optimizer.zero_grad()

        anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']
        batch, channel, width, height = anchor.shape

        pos_anchor = torch.cat([torch.ones((batch, 1, width, height), device=anchor.device), anchor], dim=1) 
        pos_pos = torch.cat([torch.zeros((batch, 1, width, height), device=pos.device), pos], dim=1)
        pos_neg = torch.cat([torch.zeros((batch, 1, width, height), device=neg.device), neg], dim=1)
        neg_anchor = torch.cat([torch.ones((batch, 1, width, height), device=pos.device), pos], dim=1) 
        neg_neg = torch.cat([torch.zeros((batch, 1, width, height), device=anchor.device), anchor], dim=1)

        pos_anchor = pos_anchor.to(device)
        pos_pos = pos_pos.to(device)
        pos_neg = pos_neg.to(device)
        neg_anchor = neg_anchor.to(device)
        neg_neg = neg_neg.to(device)


        pos_anchor_embeddings = model(pos_anchor)
        pos_pos_embeddings = model(pos_pos)
        pos_neg_embeddings = model(pos_neg)
        neg_anchor_embeddings = model(neg_anchor)
        neg_neg_embedding = model(neg_neg)

        loss, accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std = loss_fn(pos_anchor_embeddings, pos_pos_embeddings, pos_neg_embeddings, neg_anchor_embeddings, neg_neg_embedding)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss_train += loss_val
        anchor_pos_total_mean += anchor_pos_mean
        anchor_neg_total_mean += anchor_neg_mean
        total_accuracy += accuracy

        train_losses.append(total_loss_train)
        

    total_loss_train /= len(train_dataset)
    anchor_pos_mean /= len(train_dataset)
    anchor_neg_mean /= len(train_dataset)
    anchor_pos_std /= len(train_dataset)
    anchor_neg_std /= len(train_dataset)
    total_accuracy /= len(train_dataset)

    return total_loss_train, total_accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std, train_losses

def train_binary_classifier(classifier_model, embedding_model, optimizer, criterion, training_data, validation_data, epochs, device='cpu'):
    embedding_model.eval()
    # embedding_model.acrivate
    classifier_model.train()

    train_acc = []
    validation_acc = []
    

    # cache = []
    # for batch in tqdm(training_data):
    #     cache.append(batch)

    for epoch in range(epochs):
        train_acc_temp = .0
        train_loss_temp = .0
        counter = 1e-10

        for batch in tqdm(training_data):
            optimizer.zero_grad()

            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            batch_size, _, _, _ = anchor.shape

            anchor_embeddings = embedding_model(anchor)
            pos_embeddings = embedding_model(pos)
            neg_embeddings = embedding_model(neg)
            
            pos_predictions = classifier_model(anchor_embeddings -  pos_embeddings)
            neg_predictions = classifier_model(anchor_embeddings -  neg_embeddings)
            
            loss_term_1 = criterion(pos_predictions, torch.ones_like(pos_predictions))
            loss_term_2 = criterion(neg_predictions, torch.zeros_like(neg_predictions))
            loss = loss_term_1 + loss_term_2

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                posetives = torch.sum(pos_predictions > 0)
                negatives = torch.sum(neg_predictions <= 0)
                train_acc_temp += (posetives + negatives) / (batch_size * 2)
                train_loss_temp += loss.item()
            

        train_acc.append(train_acc_temp / len(training_data))
        print(f'#Epoch {epoch + 1}  Loss: {train_loss_temp}, Train accuracy: {train_acc[-1] * 100}')

        classifier_model.eval()
        with torch.no_grad():
            validation_acc_temp = .0
            validation_loss_temp = .0

            for batch in tqdm(validation_data):
                anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                batch_size, _, _, _ = anchor.shape
                anchor_embeddings = embedding_model(anchor)
                pos_embeddings = embedding_model(pos)
                neg_embeddings = embedding_model(neg)

                pos_predictions = classifier_model(anchor_embeddings -  pos_embeddings)
                neg_predictions = classifier_model(anchor_embeddings -  neg_embeddings)

                loss_term_1 = criterion(pos_predictions, torch.ones_like(pos_predictions))
                loss_term_2 = criterion(neg_predictions, torch.zeros_like(neg_predictions))
                loss = loss_term_1 + loss_term_2

                posetives = torch.sum(pos_predictions > 0)
                negatives = torch.sum(neg_predictions <= 0)
                validation_acc_temp += (posetives + negatives) / (batch_size * 2)
                validation_loss_temp += loss.item()

            validation_acc.append(validation_acc_temp / len(validation_data))
            print(f'Loss {validation_loss_temp}, validation accuracy: {validation_acc[-1] * 100}, ')
            torch.save(classifier_model.state_dict(), f'classifier_model_{epoch + 1}.pth')

    return train_acc, validation_acc

def train_mixed_image_network(embedding_model, optimizer, criterion, training_data, validation_data, epochs, device='cpu'):
    embedding_model.eval()

    train_acc = []
    validation_acc = []
    
    for epoch in range(epochs):
        train_acc_temp = .0
        train_loss_temp = .0
        counter = 1e-10

        for batch in tqdm(training_data):
            optimizer.zero_grad()

            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            batch_size, _, _, _ = anchor.shape
            anchor_pos_embeddings = embedding_model(torch.cat([anchor, pos], dim=1))
            anchor_neg_embeddings = embedding_model(torch.cat([anchor, neg], dim=1))
            
            loss, accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std = criterion(anchor_pos_embeddings, anchor_neg_embeddings)
            loss.backward()
            optimizer.step()

            train_acc_temp += accuracy
            train_loss_temp += loss.item()
            

        train_acc.append(train_acc_temp / len(training_data))
        print(f'#Epoch {epoch + 1}  Loss: {train_loss_temp}, Train accuracy: {train_acc[-1] * 100}')

        with torch.no_grad():
            validation_acc_temp = .0
            validation_loss_temp = .0

            for batch in tqdm(validation_data):
                anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)

                batch_size, _, _, _ = anchor.shape

                anchor_pos_embeddings = embedding_model(torch.cat([anchor, pos], dim=1))
                anchor_neg_embeddings = embedding_model(torch.cat([anchor, neg], dim=1))
                
                loss, accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std = criterion(anchor_pos_embeddings, anchor_neg_embeddings)

                validation_acc_temp += accuracy
                validation_loss_temp += loss.item()

            validation_acc.append(validation_acc_temp / len(validation_data))
            print(f'Loss {validation_loss_temp}, validation accuracy: {validation_acc[-1] * 100}, ')
            torch.save(embedding_model.state_dict(), f'embedding_model_{epoch + 1}.pth')

    return train_acc, validation_acc

def validate_augmented_model(validation_dataset, model, optimizer, loss_fn, device):
    model.eval()
    with torch.no_grad():
        total_loss_validation = .0
        anchor_pos_total_mean = .0
        anchor_neg_total_mean = .0
        anchor_pos_total_std = .0
        anchor_neg_total_std = .0
        total_accuracy = .0

        validation_losses = []

        for batch in tqdm(validation_dataset):
            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            batch, channel, width, height = anchor.shape
            anchor = torch.cat([torch.ones((batch, 1, width, height), device=anchor.device), anchor], dim=1)
            pos = torch.cat([torch.zeros((batch, 1, width, height), device=pos.device), pos], dim=1)
            neg = torch.cat([torch.zeros((batch, 1, width, height), device=neg.device), neg], dim=1)

            anchor_embeddings = model(anchor)
            pos_embeddings = model(pos)
            neg_embeddings = model(neg)

            loss, accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std = loss_fn(anchor_embeddings, pos_embeddings, neg_embeddings, anchor_embeddings, pos_embeddings)
            total_loss_validation += loss.item()
            anchor_pos_total_mean += anchor_pos_mean
            anchor_neg_total_mean += anchor_neg_mean
            anchor_pos_total_std += anchor_pos_std
            anchor_neg_total_std += anchor_neg_std
            total_accuracy += accuracy

            validation_losses.append(total_loss_validation)

        total_loss_validation /= len(validation_dataset)
        anchor_pos_mean /= len(validation_dataset)
        anchor_neg_mean /= len(validation_dataset)
        anchor_pos_std /= len(validation_dataset)
        anchor_neg_std /= len(validation_dataset)
        total_accuracy /= len(validation_dataset)

        return total_loss_validation, total_accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std, validation_losses

def validate_model(validation_dataset, model, optimizer, loss_fn, device):
    logs = {}
    model.eval()
    with torch.no_grad():
        total_loss_validation = .0
        anchor_pos_total_mean = .0
        anchor_neg_total_mean = .0
        anchor_pos_total_std = .0
        anchor_neg_total_std = .0
        total_accuracy = .0

        validation_losses = []

        for batch in tqdm(validation_dataset):
            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchor_embeddings = model(anchor)
            pos_embeddings = model(pos)
            neg_embeddings = model(neg)

            loss, accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std = loss_fn(anchor_embeddings, pos_embeddings, neg_embeddings)
            total_loss_validation += loss.item()
            anchor_pos_total_mean += anchor_pos_mean
            anchor_neg_total_mean += anchor_neg_mean
            anchor_pos_total_std += anchor_pos_std
            anchor_neg_total_std += anchor_neg_std
            total_accuracy += accuracy
            # print(total_accuracy, len(validation_dataset))

            validation_losses.append(total_loss_validation)

        total_loss_validation /= len(validation_dataset)
        anchor_pos_mean /= len(validation_dataset)
        anchor_neg_mean /= len(validation_dataset)
        anchor_pos_std /= len(validation_dataset)
        anchor_neg_std /= len(validation_dataset)
        total_accuracy /= len(validation_dataset)

        return total_loss_validation, total_accuracy, anchor_pos_mean, anchor_neg_mean, anchor_pos_std, anchor_neg_std, validation_losses

def train_model(train_dataset, validation_dataset, model, loss_fn, optimizer, training_step=training_step, validation_step=validate_model, epochs=10, device='cpu'):
    # fig, ax = plt.subplots(1, 1)
    train_losses = []
    validation_losses = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(1, epochs+1):
        total_loss_train, train_accuracy_value, anchor_pos_total_mean, anchor_neg_total_mean, anchor_pos_total_std, anchor_neg_total_std, train_losses_for_epoch = training_step(train_dataset, model, optimizer, loss_fn, device)
        print(f'#Epoch {epoch} loss: {total_loss_train} accuracy: {train_accuracy_value.item() * 100}, (a,p).mean: {anchor_pos_total_mean}, (a,p).std: {anchor_pos_total_std}, (a,n).mean(): {anchor_neg_total_mean}, (a,n).std: {anchor_neg_total_std}')
        
        total_loss_validation, val_accuracy_value, anchor_pos_total_mean, anchor_neg_toal_mean, anchor_pos_total_std, anchor_neg_total_std, validation_losses_for_epoch = validation_step(validation_dataset, model, optimizer, loss_fn, device)
        print(f'validation loss: {total_loss_validation} accuracy: {val_accuracy_value * 100}, (a,p).mean: {anchor_pos_total_mean}, (a,p).std: {anchor_pos_total_std}, (a,n).mean(): {anchor_neg_total_mean}, (a,n).std: {anchor_neg_total_std}')

        train_losses.extend(train_losses_for_epoch)
        validation_losses.extend(validation_losses_for_epoch)
        train_accuracy.append(train_accuracy_value.item())
        val_accuracy.append(val_accuracy_value.item())
            
        torch.save(model.state_dict(), f'embedding_model_{epoch}.pth')
        

    return train_losses, validation_losses, train_accuracy, val_accuracy

def initiate_dataset(csv_path_1, csv_path_2, dataset_code, transform=True, model='base'):
    if transform:
        if model == 'base':
            transformations = transforms.Compose([
                ToTensor(),
                Normalize(),
                # Augment()
            ])
        elif model == 'mobilenet' or model == 'vgg':
            # print('Here')
            transformations = transforms.Compose([
                ToTensor(),
                Resize(),
                Normalize(),
                # Augment()
            ])
    else:
        if model == 'base':
            transformations = transformations = transforms.Compose([
                ToTensor(),
                Normalize(),
            ])
        elif model == 'mobilenet' or model == 'vgg':
            # print('Here')
            transformations = transformations = transforms.Compose([
                ToTensor(),
                Resize(),
                Normalize(),
            ])

    kinfacei = KinFaceDataset(csv_path=csv_path_1, transform=transformations)
    kinfaceii = KinFaceDataset(csv_path=csv_path_2, transform=transformations)

    concatenated_dataset = torch.utils.data.ConcatDataset([kinfacei, kinfaceii])

    if dataset_code == 'kfi':
        return kinfacei
    elif dataset_code == 'kfii':
        return kinfaceii
    else:
        return concatenated_dataset

def load_splited_dataset(data_portion=-1, val_portion=-1, train_batch_size=128, validation_batch_size=128, test_batch_size=128, dataset_code='mix', model='base'): #=-1, validation_split=.05, train_batch_size=256, validation_batch_size=256):
    train_concatinated_data = initiate_dataset(TrainKINFaceWI, TrainKINFaceWII, dataset_code, model)
    validation_concatinated_data = initiate_dataset(ValidationKINFaceWI, ValidationKINFaceWII, dataset_code, False, model)

    if data_portion != -1:
        idx = torch.randperm(len(train_concatinated_data))[:data_portion]
        train_concatinated_data = torch.utils.data.Subset(train_concatinated_data, idx)

    if val_portion != -1:
        idx = torch.randperm(len(validation_concatinated_data))[:val_portion]
        validation_concatinated_data = torch.utils.data.Subset(validation_concatinated_data, idx)

    split_idx = len(train_concatinated_data) // 2
    rand_perm = torch.randperm(len(train_concatinated_data))
    embedding_idx = rand_perm[:split_idx]
    classification_idx = rand_perm[split_idx:]

    embedding_dataset = torch.utils.data.Subset(train_concatinated_data, embedding_idx)
    # train_dataloader = torch.utils.data.Subset(train_concatinated_data, classification_idx)

    embedding_dataloader = DataLoader(embedding_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
    classification_dataloader = DataLoader(train_concatinated_data, batch_size=train_batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
    validation_dataloader = DataLoader(validation_concatinated_data, batch_size=validation_batch_size, shuffle=True, num_workers=2, prefetch_factor=2)

    return embedding_dataloader, classification_dataloader, validation_dataloader

def load_dataset(data_portion=-1, val_portion=-1, train_batch_size=16, validation_batch_size=16, test_batch_size=16, dataset_code='mix', model='base'): #=-1, validation_split=.05, train_batch_size=256, validation_batch_size=256):
    train_concatinated_data = initiate_dataset(TrainKINFaceWI, TrainKINFaceWII, dataset_code, model=model)
    validation_concatinated_data = initiate_dataset(ValidationKINFaceWI, ValidationKINFaceWII, dataset_code, model=model)

    if data_portion != -1:
        idx = torch.randperm(len(train_concatinated_data))[:data_portion]
        train_concatinated_data = torch.utils.data.Subset(train_concatinated_data, idx)

    if val_portion != -1:
        idx = torch.randperm(len(validation_concatinated_data))[:val_portion]
        validation_concatinated_data = torch.utils.data.Subset(validation_concatinated_data, idx)

    train_dataloader = DataLoader(train_concatinated_data, batch_size=train_batch_size, shuffle=True, num_workers=2, prefetch_factor=2)
    validation_dataloader = DataLoader(validation_concatinated_data, batch_size=validation_batch_size, shuffle=True, num_workers=2, prefetch_factor=2)

    return train_dataloader, validation_dataloader

def validate_model(embedding_model, classifier_model, dataset, device='cpu'):
    sim = torch.nn.CosineSimilarity()
    embedding_model.eval()

    if classifier_model is not None:
        classifier_model.eval()
    
    accuracy = .0
    for batch in tqdm(dataset):
        anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor_embeddings = embedding_model(anchor)
        pos_embeddings = embedding_model(pos)
        neg_embeddings = embedding_model(neg)

        if classifier_model is not None:
            posetives = classifier_model(torch.cat((anchor_embeddings, pos_embeddings), dim=1))
            negatives = classifier_model(torch.cat((anchor_embeddings, neg_embeddings), dim=1))
            # print(posetives)
            # print(negatives)
            posetives_count = torch.sum(posetives > 0)
            negatives_count = torch.sum(negatives <= 0)

            batch, _, _, _ = anchor.shape

            accuracy += (posetives_count + negatives_count) / (batch * 2)
        else:
            count, _, _, _ = anchor.shape
            anchor_pos_diff = 1 - sim(anchor_embeddings, pos_embeddings)
            anchor_neg_diff = 1 - sim(anchor_embeddings, neg_embeddings)

            threshold = anchor_pos_diff.mean() + anchor_pos_diff.std() 
            pos_count = torch.sum(anchor_pos_diff <= threshold)
            neg_count = torch.sum(anchor_neg_diff > threshold)

            accuracy += (pos_count + neg_count) / (count * 2)
            # print((pos_count + neg_count) / (count * 2))
            # print(acuuracy)
            
    
    accuracy /= len(dataset)
    return accuracy

def log_model(model, dataset, alpha, device='cpu'):
    model.eval()
    sim = torch.nn.CosineSimilarity()

    anchor_pos_wrongs = {
            'anchor': [],
            'pos': []
        }

    anchor_neg_wrongs = {
        'anchor': [],
        'neg': []
    }

    accuracy = .0
    for batch in tqdm(dataset):
        anchor, pos, neg, metadata = batch['anchor'], batch['pos'], batch['neg'], batch['index']

        anchor = anchor.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        batch_size, _, _, _ = anchor.shape
        # print(batch_size)
        anchor_embeddings = model(anchor)
        pos_embeddings = model(pos)
        neg_embeddings = model(neg)

        anchor_pos_diff = 1 - sim(anchor_embeddings, pos_embeddings)
        anchor_neg_diff = 1 - sim(anchor_embeddings, neg_embeddings)

        threshold = anchor_pos_diff.mean() + anchor_pos_diff.std() 
        pos_mask = anchor_pos_diff <= threshold
        neg_mask = anchor_neg_diff > threshold
        pos_count = torch.sum(pos_mask)
        neg_count = torch.sum(neg_mask)

        accuracy += (pos_count + neg_count) / (batch_size * 2)
        # print(pos_mask.shape)
        # print(pos_mask)
        # print(metadata['parent'])
        # print(metadata['parent'].shape)
        # print(pos_count, neg_count)
        anchor_pos_wrongs['anchor'].extend(np.array(metadata['parent'])[~(pos_mask.cpu().numpy())])
        anchor_pos_wrongs['pos'].extend(np.array(metadata['child'])[~(pos_mask.cpu().numpy())])
        
        anchor_neg_wrongs['anchor'].extend(np.array(metadata['parent'])[~(neg_mask.cpu().numpy())])
        anchor_neg_wrongs['neg'].extend(np.array(metadata['negative_child'])[~(neg_mask.cpu().numpy())])

    accuracy /= len(dataset)
    
    anchor_pos_df = pd.DataFrame.from_dict(anchor_pos_wrongs)
    anchor_neg_df = pd.DataFrame.from_dict(anchor_neg_wrongs)

    anchor_pos_df.to_csv('anchor_pos_df.csv')
    anchor_neg_df.to_csv('anchor_neg_df.csv')

    return accuracy, anchor_pos_wrongs, anchor_neg_wrongs      

def generate_embeddings(embedding_model, dataset, device='cpu'):
    embedding_model.eval()

    anchor_embeddings_list = []
    pos_embeddings_list = []
    neg_embeddings_list = []

    accuracy = .0
    with torch.no_grad():
        for batch in tqdm(dataset):
            anchor, pos, neg = batch['anchor'], batch['pos'], batch['neg']

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchor_embeddings = embedding_model(anchor)
            pos_embeddings = embedding_model(pos)
            neg_embeddings = embedding_model(neg)

            anchor_embeddings_list.extend(anchor_embeddings.cpu().numpy())
            pos_embeddings_list.extend(pos_embeddings.cpu().numpy())
            neg_embeddings_list.extend(neg_embeddings.cpu().numpy())
    # print(embeddings[0].shape)
    # print(embeddings[1].shape)

    return {
        'anchor': anchor_embeddings_list,
        'pos': pos_embeddings_list,
        'neg': neg_embeddings_list
    }