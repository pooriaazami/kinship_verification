import os
import itertools

BASE_PATH = 'data\\KinFaceW\\'

KinFaceWI = 'KinFaceW-I\\'
KinFaceWII = 'KinFaceW-II\\'

def load_pairs(dataset_path):
    pairs = []
    for root, _, files in os.walk(dataset_path):
        index = 0
        while index < len(files) - 1:
            pairs.append(
                {'id': len(pairs), 'root': root, 'parent': files[index], 'child': files[index + 1]})
            index += 2

    return pairs

def generate_priplets(pairs):
    triplets = []

    for first, second in itertools.product(pairs, pairs):
        if first['id'] == second['id']:
            continue

        triplets.append({
            'id': len(triplets),
            'parent': first['root'] + first['parent'],
            'child': first['root'] + first['child'],
            'negative_child': second['root'] + second['parent']
        })

        triplets.append({
            'id': len(triplets),
            'parent': first['root'] + first['parent'],
            'child': first['root'] + first['child'],
            'negative_child': second['root'] + second['child']
        })

    return triplets


kinface_pairs_1 = load_pairs(BASE_PATH + KinFaceWI)
kinface_pairs_2 = load_pairs(BASE_PATH + KinFaceWII)

kinface_triplets_1 = generate_priplets(kinface_pairs_1)
kinface_triplets_2 = generate_priplets(kinface_pairs_2)
# print(len(kinface_triplets_1) + len(kinface_triplets_2))

# print(kinface_triplets_1[125])
