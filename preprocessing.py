import numpy as np
from matplotlib import pyplot as plt

from dataset import IMAGE_SIZE, read_dataset, write_dataset
from merge_results import meanWeightedAnal

MAX_CENTER_SIZE = 5


def extract_data(datapoint):
    """
    :param datapoint: A datapoint, as it appears in the original dataset
    :return: band1 and band2, reshaped into proper images
    """
    band1 = np.asarray(datapoint['band_1']).reshape(IMAGE_SIZE, IMAGE_SIZE)
    band2 = np.asarray(datapoint['band_2']).reshape(IMAGE_SIZE, IMAGE_SIZE)
    return [band1, band2]


def fourier_and_reverse(band, center_size):
    """
    Perform fourier analysis on the given band

    :param band: 75x75 numpy array
    :param center_size:
    :return:
    """
    f = np.fft.fft2(band)
    fshift = np.fft.fftshift(f)
    rows, cols = band.shape
    crow, ccol = round(rows / 2), round(cols / 2)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    fshift[crow - center_size:crow + center_size, ccol - center_size:ccol + center_size] = 0
    #plt.imshow(magnitude_spectrum, cmap='gray')
    #plt.show()
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def remove_noise(center_size=1, bands=None):
    """
    Removes noise from the given images
    """
    reversed_band_1 = fourier_and_reverse(bands[0], center_size)
    reversed_band_2 = fourier_and_reverse(bands[1], center_size)
    return [reversed_band_1, reversed_band_2]


def nabla(band):
    derivates =  np.gradient(band)
    # arrx = band.convolve2d(band, xder, mode='valid')
    # arry = band.convolve2d(band, yder, mode='valid')
    return np.hypot(derivates[0], derivates[1])


def scale(images):
    """
    Performs normalization and scaling, so that all values are in the range [0, 1].
    Warning: to avoid OOM errors, will mutate the input list contents instead of returning a new array

    :param images: list of a list of the two bands (each 75x75)
    :return: The input param images
    """
    minVal, maxVal = 40, -40
    for band1, band2 in images:
        minVal = min(np.min(band1), np.min(band2), minVal)
        maxVal = max(np.max(band1), np.max(band2), maxVal)

    for image in images:
        image[0] = (image[0] - minVal) / maxVal
        image[1] = (image[1] - minVal) / maxVal

    return images


def prep_dataset(data):
    """
    Replaces the band_1 and band_2 values with "noiseless" and normalized values in the input data.
    Warning: makes changes in-place.

    :param data: raw dataset from Kaggle
    :return: Same dataset
    """
    images = [None] * len(data)
    # Fourier
    for i in range(len(data)):
        images[i] = remove_noise(bands=extract_data(data[i]))  # Two matrices of dimensions 75x75

    # Scale
    scale(images)

    for i in range(len(data)):
        # original_bands = extract_data(data[i])
        title = 'iceberg' if data[i]['is_iceberg'] else 'ship'
        # plt.subplot(231), plt.title('Original HH ' + title), plt.imshow(original_bands[0], cmap='gray')
        # plt.subplot(234), plt.title('Original HV ' + title), plt.imshow(original_bands[1], cmap='gray')
        data[i]['band_1'] = images[i][0].flatten().tolist()
        data[i]['band_2'] = images[i][1].flatten().tolist()
        added = np.add(images[i][0], images[i][1])
        # data[i]['band_add'] = added.flatten().tolist()
        data[i]['band_nabla'] = nabla(added).flatten().tolist()

        # title = 'iceberg' if data[i]['is_iceberg'] else 'ship'
        plt.subplot(321), plt.title('Fourier HH '+title), plt.imshow(images[i][0], cmap='gray')
        plt.subplot(322), plt.title('Fourier HV'), plt.imshow(images[i][1], cmap='gray')
        plt.subplot(323), plt.title('Added fourier'), plt.imshow(added, cmap='gray')
        plt.subplot(324), plt.title('Nabla'), plt.imshow(nabla(added), cmap='gray')
        plt.show()

    return data


def main():
    data = read_dataset('train.json')
    data = prep_dataset(data)
    write_dataset('train_processed.json', data)
    data = None # Use less memory
    data = prep_dataset(read_dataset('test.json'))

    chunk_size = len(data) // 10
    for i in range(10):
        if i < 9:
            print('Saving datapoints [', i * chunk_size, ':', (i + 1) * chunk_size, ']')
            write_dataset('test_processed_' + str(i) + '.json', data[i * chunk_size:(i + 1) * chunk_size])
        else:
            print('Saving datapoints [', i * chunk_size, ':', len(data), ']')
            write_dataset('test_processed_' + str(i) + '.json', data[i * chunk_size:])
    print('Saving entire dataset', len(data))
    write_dataset('test_processed_full.json', data)


def nabla_analyze(data):
    """
    Experimenting with nabla

    :param data:
    :return:
    """
    original = extract_data(data)
    original.append(np.add(original[0], original[1]))

    fourier = [fourier_and_reverse(band, 1) for band in original]
    nabla_list = [nabla(fourier[0]), nabla(fourier[1]), nabla(fourier[2]),nabla(np.add(fourier[0], fourier[1]))]
    # nabla_first = [fourier_and_reverse(nabla(band), 1) for band in original]
    '''
    plt.subplot(221)
    plt.title('HH ' + ('iceberg' if data['is_iceberg'] else 'ship'))
    plt.imshow(original[0], cmap='gray')
    plt.subplot(222)
    plt.title('HV '+ ('iceberg' if data['is_iceberg'] else 'ship'))
    plt.imshow(original[1], cmap='gray')
    plt.subplot(223)
    plt.title('HH filtered')
    plt.imshow(fourier[0], cmap='gray')
    plt.subplot(224)
    plt.title('HV filtered')
    plt.imshow(fourier[1], cmap='gray')

    
    plt.subplot(121)
    plt.title('HH filtered')
    plt.imshow(fourier[0], cmap='gray')
    plt.subplot(122)
    plt.title('HV filtered')
    plt.imshow(fourier[1], cmap='gray')
    plt.suptitle('Noise-filtered and gradient image of ' + ('iceberg' if data['is_iceberg'] else 'ship'), size=20)
    plt.subplot(223)
    '''
    plt.suptitle('Gradient, image of '+ ('iceberg' if data['is_iceberg'] else 'ship'), size=20)
    plt.imshow(nabla_list[2], cmap='gray')
    plt.show()
    '''
    plt.show()
    #plt.subplot(333)
    #plt.title('HH + HV')
    #plt.imshow(original[2], cmap='gray')
    #plt.subplot(334)
    #plt.title('HH fourier->nabla')
    #plt.imshow(nabla_list[0], cmap='gray')
    #plt.subplot(335)
    #plt.title('HV fourier->nabla')
    #plt.imshow(nabla_list[1], cmap='gray')
    #plt.subplot(336)
    #plt.title('HH+HV->fourier->nabla')
    #plt.imshow(nabla_list[2], cmap='gray')
    #plt.subplot(337)
    #plt.title('fourier->HH+HV->nabla')
    #plt.imshow(nabla_list[2], cmap='gray')
    # plt.subplot(337)
    # plt.title('HH nabla->fourier')
    # plt.imshow(nabla_first[0], cmap='gray')
    # plt.subplot(338)
    # plt.title('HV nabla->fourier')
    # plt.imshow(nabla_first[1], cmap='gray')
    # plt.subplot(339)
    # plt.title('HH + HV nabla->fourier')
    # plt.imshow(nabla_first[2], cmap='gray')
    #plt.show()
    '''
def fourier_analyze(datapoint, center_size):
    """
    Used to experiment with and debugging the fourier analysis
    """
    band1, band2 = extract_data(datapoint)

    reversed_band_1 = fourier_and_reverse(band1, center_size)
    reversed_band_2 = fourier_and_reverse(band2, center_size)

    plt.subplot(141), plt.imshow(band1, cmap='gray')
    plt.title('HH iceberg' if datapoint['is_iceberg'] else 'HH ship'), plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(reversed_band_1, cmap='gray')
    plt.title('Reverse HH'), plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(band2, cmap='gray')
    plt.title('HV iceberg' if datapoint['is_iceberg'] else 'HV ship'), plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(reversed_band_2, cmap='gray')
    plt.title('Reverse HV'), plt.xticks([]), plt.yticks([])
    plt.show()
    return np.mean(reversed_band_1), np.max(reversed_band_1)

def max_mean_noise(data):
    """
    Code used to determine center size for fourier
    """
    for datapoint in data:
        print(datapoint['id'], datapoint['is_iceberg'])
        max_list = np.zeros(MAX_CENTER_SIZE)
        mean_list = np.zeros(MAX_CENTER_SIZE)
        filter_list = np.arange(0, MAX_CENTER_SIZE, 1)
        # for i in range(MAX_CENTER_SIZE):
        mean_list[1], max_list[1] = fourier_analyze(datapoint, 1)
        plt.plot(filter_list, max_list, color='navy', label='Max intensity sigma as function of filtersize')
        plt.plot(filter_list, mean_list, color='magenta', label='Mean intensity sigma as function of filtersize')
        # plt.show()


if __name__ == '__main__':
    dataset = read_dataset('train.json')
    for datapoint in dataset:
        nabla_analyze(datapoint)

