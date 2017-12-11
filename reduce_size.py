import numpy as np
from matplotlib import pyplot as plt

from dataset import read_dataset, write_dataset, IMAGE_SIZE


def stupid_crop(band, new_size):
    m = IMAGE_SIZE // 2 - (new_size // 2)
    return band[m:m + new_size, m:m + new_size]


def stupid_crop_data(dataset):
    for i in range(len(dataset)):
        for band_name in ['band_1', 'band_2']:#, 'band_nabla']:
            band = np.asarray(dataset[i][band_name]).reshape(IMAGE_SIZE, IMAGE_SIZE)
            cropped = stupid_crop(band, int(IMAGE_SIZE / 1.5))
            dataset[i][band_name] = cropped.flatten().tolist()
            # plt.subplot(121)
            # plt.title(str(band.shape))
            # plt.imshow(band, cmap='gray')
            # plt.subplot(122)
            # plt.title(str(cropped.shape))
            # plt.imshow(cropped, cmap='gray')
            # plt.show()
    return dataset


def max_mean(dataset):
    maxMean = 0
    id = 'hola_senor'
    max_i = 0
    for i in range(len(dataset)):
        newMean = np.mean(np.asarray(dataset[i]['band_1']).reshape(IMAGE_SIZE, IMAGE_SIZE))
        if newMean > maxMean:
            maxMean = newMean
            id = dataset[i]['id']
            max_i = i

    band = np.asarray(dataset[max_i]['band_1']).reshape(IMAGE_SIZE, IMAGE_SIZE)
    plt.imshow(stupid_crop(band, int(IMAGE_SIZE // 1.5)), cmap='gray')
    plt.show()
    print(id)


def main():
    # for file in [
    #     'train_processed',
    #     'test_processed_0',
    #     'test_processed_1',
    #     'test_processed_2',
    #     'test_processed_3',
    #     'test_processed_4',
    #     'test_processed_5',
    #     'test_processed_6',
    #     'test_processed_7',
    #     'test_processed_8',
    #     'test_processed_9',
    #     'test_processed_full'
    # ]:
    for file in [
        'train',
        'test'
    ]:
        dataset = read_dataset(file + '.json')
        # max_mean(dataset)
        stupid_crop_data(dataset)
        write_dataset(file + '_cropped.json', dataset)


def debug():
    dataset = read_dataset('train_processed.json')
    max_mean(dataset)
    for i in range(len(dataset)):
        bands = np.array([])
        for name in ['band_1', 'band_2', 'band_nabla']:
            bands = np.append(bands, np.asarray(dataset[i][name]))
        dataset[i] = bands


if __name__ == '__main__':
    # debug()
    main()
