import os
import json
import numpy as np

IMAGE_SIZE = 75
IMAGE_DIR = os.getcwd() + "/images/"


def read_dataset(file='train.json'):
    with open(file) as f:
        print('Reading', file)
        data = json.load(f)
        print('Dataset size', len(data))
    return data


def write_dataset(file, data):
    print('Writing to', file)
    with open(file, 'w') as f:
        json.dump(data, f, separators=(',', ':'))


def write_response(ids, results):
    f = open('results.csv', 'w')
    f.write('id,is_iceberg\n')
    for i in range(len(ids)):
        f.write(str(ids[i]) + ',' + str(results[i][0]) + '\n')
    f.close()


def visualize(data):
    from PIL import Image
    for item in data:
        # print(item['id'], item['is_iceberg'], item['inc_angle'])
        band1 = np.asarray(item['band_1']).reshape(IMAGE_SIZE, IMAGE_SIZE)
        band2 = np.asarray(item['band_2']).reshape(IMAGE_SIZE, IMAGE_SIZE)
        # total_max = max(total_max, np.max(band2))
        # total_min = min(total_min, np.min(band2))
        scaled1 = ((band1 - 46) * (255.0 / 35.0))
        scaled2 = ((band2 - 46) * (255.0 / 21.0))
        Image.fromarray(scaled1.astype(np.uint8)).save(
            IMAGE_DIR + ('ice_' if item['is_iceberg'] else 'ship_') + item['id'] + '_1.jpg'
        )
        Image.fromarray(scaled2.astype(np.uint8)).save(
            IMAGE_DIR + ('ice_' if item['is_iceberg'] else 'ship_') + item['id'] + '_2.jpg'
        )


def check_last_100(data):
    for item in data[-100:-1]:
        print(item['is_iceberg'])
        # visualize(read_dataset())


if __name__ == '__main__':
    visualize(read_dataset('train.json'))