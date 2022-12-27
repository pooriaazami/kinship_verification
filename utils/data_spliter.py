import os
import random
import itertools

import pandas as pd

BASE_PATH = 'data\\KinFaceW\\'

KinFaceWI = 'KinFaceW-I\\'
KinFaceWII = 'KinFaceW-II\\'

def read_folds(folds_path, folds):
    images = []

    with open(folds_path) as file:
        for index, line in enumerate(file):
            if index + 1 not in folds:
                continue

            line = line.strip()
            for item in line.split(' '):
                images.append(item)

    return images
        

def get_images(source):
    pairs = {}
    count = 0

    for root, _, files in os.walk(source):
        if len(files) % 2 != 0:
            files.pop()

        if len(files) > 4:
            pairs[root] = files
            count += len(files)
    
    count //= 2

    return pairs, count

def load_pairs(dataset_path, validation_split, test_split):
    print(f'[log]: Loading triplets from {dataset_path}')
    train_pairs = []
    validation_pairs = []
    test_pairs = []

    counter = 0
    validation_split += test_split
    # print(validation_split)
    for root, _, files in os.walk(dataset_path):
        index = 0
        counter += 1
        print(files)
        while index < len(files) - 1:
            sample = random.random()
            if sample <= test_split:
                test_pairs.append({'id': counter, 'root': root, 'parent': files[index], 'child': files[index + 1]})
            elif sample <= validation_split:
                validation_pairs.append({'id': counter, 'root': root, 'parent': files[index], 'child': files[index + 1]})
            else:
                train_pairs.append({'id': counter, 'root': root, 'parent': files[index], 'child': files[index + 1]})

            index += 2

    return train_pairs, validation_pairs, test_pairs

def load_pairs_from_folds(dataset_path, allow_images):
    allow_images = set(allow_images)
    print(f'[log]: Loading triplets from {dataset_path}')
    pairs = []
    for root, _, files in os.walk(dataset_path):
        index = 0
        while index < len(files) - 1:
            if files[index] in allow_images and files[index + 1] in allow_images:
                pairs.append(
                    {'id': len(pairs), 'root': root, 'parent': files[index], 'child': files[index + 1]})
            index += 2

    return pairs

def generate_priplets(pairs):
    print(f'[log]: generating triples...')
    triplets = []

    for first, second in itertools.product(pairs, pairs):
        if first['id'] == second['id']:
            continue

        triplets.append({
            'id': len(triplets),
            'parent': first['root'] + '\\' + first['parent'],
            'child': first['root'] + '\\' + first['child'],
            'negative_child': second['root'] + '\\' + second['parent']
        })

        triplets.append({
            'id': len(triplets),
            'parent': first['root'] + '\\' + first['parent'],
            'child': first['root'] + '\\' +first['child'],
            'negative_child': second['root'] + '\\' +  second['child']
        })

    return triplets

def generate_csv_file(dataset_path, csv_path):
    print(f'[log]: generating csv file from {dataset_path}')
    pairs = load_pairs(dataset_path)
    triplets = generate_priplets(pairs)

    df = pd.DataFrame.from_dict(triplets)
    df.to_csv(csv_path, index=False)
    print('[log]: Done')

def generate_csv_file(dataset_path, csv_path, validation_split, test_split):
    print(f'[log]: generating csv file from {dataset_path}')
    train_pairs, validation_pairs, test_pairs = load_pairs(dataset_path, validation_split, test_split)

    train_triplets = generate_priplets(train_pairs)
    validation_triplets = generate_priplets(validation_pairs)
    test_triplets = generate_priplets(test_pairs)

    train_df = pd.DataFrame.from_dict(train_triplets)
    validation_df = pd.DataFrame.from_dict(validation_triplets)
    test_df = pd.DataFrame.from_dict(test_triplets)

    train_df.to_csv('Train' + csv_path, index=False)
    validation_df.to_csv('Validation' + csv_path, index=False)
    test_df.to_csv('Test' + csv_path, index=False)

    print('[log]: Done')

def generate_csv_file_from_folds(dataset_path, folds_path, csv_path, folds):
    print(f'[log]: generating csv file from {dataset_path}')
    allow_images = read_folds(folds_path, folds)
    pairs = load_pairs_from_folds(dataset_path, allow_images)

    triplets = generate_priplets(pairs)
    df = pd.DataFrame.from_dict(triplets)
    df.to_csv(csv_path, index=False)

    print('[log]: Done')


if __name__ == '__main__':
    generate_csv_file_from_folds(BASE_PATH + KinFaceWI,  'data\\KinFaceWIFolds.txt', 'KinFaceWITrainFolds.csv', [1, 2, 3, 4])
    generate_csv_file_from_folds(BASE_PATH + KinFaceWI,  'data\\KinFaceWIFolds.txt', 'KinFaceWITestFolds.csv', [5])

    generate_csv_file_from_folds(BASE_PATH + KinFaceWII,  'data\\KinFaceWIIFolds.txt', 'KinFaceWIITrainFolds.csv', [1, 2, 3, 4])
    generate_csv_file_from_folds(BASE_PATH + KinFaceWII,  'data\\KinFaceWIIFolds.txt', 'KinFaceWIITestFolds.csv', [5])
