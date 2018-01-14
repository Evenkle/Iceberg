import json
import pprint
import numpy as np
import random
from tensorflow.python import keras as ks



def load_data(devset_size=320, dims=(75, 75)):
    with open("train_processed.json") as f:
        data = json.load(f)

    random.shuffle(data)

    num_data = len(data)

    x_train = np.empty((num_data - devset_size, dims[0], dims[1], 2))
    y_train = np.empty((num_data - devset_size, 2))
    x_test = np.empty((devset_size, dims[0], dims[1], 2))
    y_test = np.empty((devset_size, 2))

    i = 0
    for item in data[0:-devset_size]:
        band1 = np.asarray(item["band_1"]).reshape((dims[0], dims[1]))
        band2 = np.asarray(item["band_2"]).reshape((dims[0], dims[1]))
        # band3 = np.asarray(item["band_3"]).reshape((dims[0],dims[1]))
        x_train[i] = np.dstack((band1, band2))
        #y_train[i] = item['is_iceberg'] == 1
        if item['is_iceberg'] == 1:
            y_train[i] = [1, 0]
        else:
            y_train[i] = [0, 1]
        i += 1

    i = 0
    for item in data[num_data - devset_size:]:
        band1 = np.asarray(item["band_1"]).reshape((dims[0], dims[1]))
        band2 = np.asarray(item["band_2"]).reshape((dims[0], dims[1]))
        # band3 = np.asarray(item["band_3"]).reshape((dims[0],dims[1]))
        x_test[i] = np.dstack((band1, band2))
        #y_test[i] = item['is_iceberg'] == 1
        if item['is_iceberg'] == 1:
            y_test[i] = [1, 0]
        else:
            y_test[i] = [0, 1]
        i += 1

    return x_train, y_train, x_test, y_test


def load_test_data(dims=(75,75)):
    with open("test_processed.json") as f:
        data = json.load(f)

    num_data = len(data)

    images = np.empty((num_data, dims[0], dims[1], 2))
    id = []
    i = 0
    for item in data:
        band1 = np.asarray(item["band_1"]).reshape((dims[0], dims[1]))
        band2 = np.asarray(item["band_2"]).reshape((dims[0], dims[1]))
        # band3 = np.asarray(item["band_3"]).reshape((dims[0],dims[1]))
        images[i] = np.dstack((band1, band2))
        id.append(item["id"])
        i += 1
    return images, id


def identity_block(X, f, filters, stage, block, seed=None):
    """
    Implementation of the identity block. A three layer residual block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = ks.layers.Conv2D(filters=F1,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='valid',
                         name=conv_name_base + '2a',
                         kernel_initializer=ks.initializers.glorot_uniform(seed))(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = ks.layers.Activation('relu')(X)

    # Second component of main path
    X = ks.layers.Conv2D(filters=F2,
                         kernel_size=(f, f),
                         strides=(1, 1),
                         padding='same',
                         name=conv_name_base + '2b',
                         kernel_initializer=ks.initializers.glorot_uniform(seed))(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = ks.layers.Activation('relu')(X)

    # Third component of main path
    X = ks.layers.Conv2D(filters=F3,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='valid',
                         name=conv_name_base + '2c',
                         kernel_initializer=ks.initializers.glorot_uniform(seed))(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = ks.layers.Add()([X, X_shortcut])
    X = ks.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2, seed=None):
    """
    Implementation of the convolutional block.

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = ks.layers.Conv2D(filters=F1,
                         kernel_size=(1, 1),
                         strides=(s, s),
                         padding='valid',
                         name=conv_name_base + '2a',
                         kernel_initializer=ks.initializers.glorot_uniform(seed))(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = ks.layers.Activation('relu')(X)

    # Second component of main path
    X = ks.layers.Conv2D(filters=F2,
                         kernel_size=(f, f),
                         strides=(1, 1),
                         padding='same',
                         name=conv_name_base + '2b',
                         kernel_initializer=ks.initializers.glorot_uniform(seed))(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = ks.layers.Activation('relu')(X)

    # Third component of main path
    X = ks.layers.Conv2D(filters=F3,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='valid',
                         name=conv_name_base + '2c',
                         kernel_initializer=ks.initializers.glorot_uniform(seed))(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # Shortcut path
    X_shortcut = ks.layers.Conv2D(filters=F3,
                                  kernel_size=(1, 1),
                                  strides=(s, s),
                                  padding='valid',
                                  name=conv_name_base + '1',
                                  kernel_initializer=ks.initializers.glorot_uniform())(X_shortcut)

    X_shortcut = ks.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = ks.layers.Add()([X, X_shortcut])
    X = ks.layers.Activation('relu')(X)

    return X


def potato_peal(input_shape=(75, 75, 2), classes=2):
    """
    The potato peal residual network. A small network with
     Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = ks.layers.Input(input_shape)

    # Zero-Padding
    X = ks.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = ks.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=ks.initializers.glorot_uniform())(
        X)
    X = ks.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = ks.layers.Activation('relu')(X)
    X = ks.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # Avgpool
    X = ks.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # output layer
    X = ks.layers.Flatten()(X)
    X = ks.layers.Dense(classes, activation='softmax', name='fc' + str(classes),
                        kernel_initializer=ks.initializers.glorot_uniform())(X)

    # Create model
    model = ks.models.Model(inputs=X_input, outputs=X, name='Potato_peal')

    return model


if __name__ == "__main__":
    print("loading the data")
    x_train, y_train, x_test, y_test = load_data()
    print("done!")

    print("building model")
    peal = potato_peal(input_shape=(75, 75, 2), classes=2)
    peal.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    print("done!")

    tests = []
    for i in range(1):
        peal.fit(x_train, y_train, 32, 1)
        scores = peal.evaluate(x_test, y_test, verbose=1)
        tests.append(scores)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    pprint.pprint(tests)

    print("saving model")
    peal.save("models/potato_peal")

    print("clearing variable")

    x_train, y_train, x_test, y_test = None, None, None, None

    print("loading test data")
    images, id = load_test_data()

    print("predicting")
    predictions = peal.predict(images)

    print("creating csv")
    with open("peal.csv", 'w') as f:
        string = 'id,is_iceberg\n'
        f.write(string)
        for i in range(len(predictions)):
            string = id[i] + "," + str(predictions[i][0]) + "\n"
            f.write(string)
