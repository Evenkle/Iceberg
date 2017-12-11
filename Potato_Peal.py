import numpy as np
import tensorflow as tf
import tensorflow.python.keras as ks


def identity_block(X, f, filters, stage, block):
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

    # Save the input value
    X_shortcut = X

    # First component of the main path
    X = ks.layers.Conv2D(filters=F1,
                         kernel_size=(1, 1),
                         strides=(1,1),
                         padding='valid',
                         name=conv_name_base + '2a',
                         kernel_initializer=ks.initializers.glorot_uniform())(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = ks.layers.Activation('relu')(X)

    # Second component of the main path
    X = ks.layers.Conv2D(filters=F2,
                         kernel_size=(f,f),
                         strides=(1,1),
                         padding='same',
                         name=conv_name_base + '2b',
                         kernel_initializer=ks.initializers.glorot_uniform())(X)

    X = ks.layers.BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = ks.layers.Activation('relu')(X)

    # Third component of the main path
    X = ks.layers.Conv2D(filters=F3,
                         kernel_size=(1,1),
                         strides=(1,1),
                         padding='valid',
                         name=conv_name_base + '2c',
                         kernel_initializer=ks.initializers.glorot_uniform())(X)

    X = ks.layers.Add()([X, X_shortcut])
    X = ks.layers.Activation('relu')(X)

    return X



