import os
import itertools

import pandas as pd

BASE_PATH = 'data\\KinFaceW\\'

KinFaceWI = 'KinFaceW-I\\'
KinFaceWII = 'KinFaceW-II\\'

def load_pairs(dataset_path):
    print(f'[log]: Loading triplets from {dataset_path}')
    pairs = []
    for root, _, files in os.walk(dataset_path):
        index = 0
        while index < len(files) - 1:
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

if __name__ == '__main__':
    generate_csv_file(BASE_PATH + KinFaceWI, 'KinFaceWITriplets.csv')
    generate_csv_file(BASE_PATH + KinFaceWII, 'KinFaceWIITriplets.csv')
