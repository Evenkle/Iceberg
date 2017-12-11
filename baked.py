import tensorflow as tf
import json
import random
import csv
import numpy as np
import matplotlib.pyplot as plt

# The names of the files that we load, its a lot of data that's why its split in so manny files
file_names = ['test_processed_0_cropped.json',
              'test_processed_1_cropped.json',
              'test_processed_2_cropped.json',
              'test_processed_3_cropped.json',
              'test_processed_4_cropped.json',
              'test_processed_5_cropped.json',
              'test_processed_6_cropped.json',
              'test_processed_7_cropped.json',
              'test_processed_8_cropped.json',
              'test_processed_9_cropped.json']

# Parameters
dev_set_size = 320

# Image dimensions
dimentions = (50, 50)
learning_rate = 1e-4
dropout_prob = 0.5

batch_size = 50
num_iter = 500
make_csv = True
print_after = False

num_images = 3

# conv1
conv1_f = 3
conv1_num_filters = 15

# conv2
conv2_f = 3
conv2_num_filters = 15

# conv3
conv3_f = 3
conv3_num_filters = 15

# conv4
conv4_f = 3
conv4_num_filters = 15

# conv5
conv5_f = 3
conv5_num_filters = 15

# conv6
conv6_f = 3
conv6_num_filters = 30

num_hidden_fc1 = 500
num_hidden_fc2 = 2


# Loading the data from the files and preparing it for the Network
def load_data():
    # Read the data
    with open('3-band-fourier-nabla/train_processed_cropped.json') as data_file:
        data = json.load(data_file)

    random.shuffle(data)

    x_train = []
    x_train_angle = []
    y_train = []

    # ?, 2500, 3

    for i in range(len(data) - dev_set_size):
        chanels = []
        for j in range(len(data[0]['band_1'])):
            chanels.append([data[i]['band_1'][j], data[i]['band_2'][j], data[i]['band_nabla'][j]])
        x_train.append(chanels)
        x_train_angle.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        if data[i]['is_iceberg'] == 0:
            y_train.append([0, 1])
        else:
            y_train.append([1, 0])

    x_test = []
    x_test_angle = []
    y_test = []
    for i in range(len(data) - dev_set_size, len(data)):
        chanels = []
        for j in range(len(data[0]['band_1'])):
            chanels.append([data[i]['band_1'][j], data[i]['band_2'][j], data[i]['band_nabla'][j]])
        x_test.append(chanels)
        x_test_angle.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        if data[i]['is_iceberg'] == 0:
            y_test.append([0, 1])
        else:
            y_test.append([1, 0])

    return x_train, x_train_angle, y_train, x_test, x_test_angle, y_test


# loading the test data and preparing it for the network
def load_test_data(file_name):
    # Read the data
    with open('3-band-fourier-nabla/'+file_name) as data_file:
        data = json.load(data_file)

    x_test = []
    x_test_angle = []
    id = []

    for i in range(len(data)):
        channels = []
        for j in range(len(data[0]['band_1'])):
            channels.append([data[i]['band_1'][j], data[i]['band_2'][j], data[i]['band_nabla'][j]])
        x_test.append(channels)
        x_test_angle.append([data[i]['inc_angle']] if data[i]['inc_angle'] != 'na' else [0])
        id.append(data[i]['id'])

    return x_test, x_test_angle, id


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_tensor(x):
    """method for making the convolution blocks of the network"""
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, dimentions[0], dimentions[1], num_images])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([conv1_f, conv1_f, num_images, conv1_num_filters])
        b_conv1 = bias_variable([conv1_num_filters])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([conv2_f, conv2_f, conv1_num_filters, conv2_num_filters])
        b_conv2 = bias_variable([conv2_num_filters])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([conv3_f, conv3_f, conv2_num_filters, conv3_num_filters])
        b_conv3 = bias_variable([conv3_num_filters])
        h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    with tf.name_scope("conv4"):
        W_conv4 = weight_variable([conv4_f, conv4_f, conv3_num_filters, conv4_num_filters])
        b_conv4 = bias_variable([conv4_num_filters])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv4)

    with tf.name_scope('conv5'):
        W_conv5 = weight_variable([conv5_f, conv5_f, conv4_num_filters, conv5_num_filters])
        b_conv5 = bias_variable([conv5_num_filters])
        h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    with tf.name_scope("conv6"):
        W_conv6 = weight_variable([conv6_f, conv6_f, conv5_num_filters, conv6_num_filters])
        b_conv6 = bias_variable([conv6_num_filters])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

    with tf.name_scope("pool3"):
        h_pool3 = max_pool_2x2(h_conv6)

    return h_pool3


def fully_connected(x, angle):
    """Method for making two fully connected layers"""
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * conv6_num_filters+1, num_hidden_fc1])
        b_fc1 = bias_variable([num_hidden_fc1])

        h_pool_flat = tf.reshape(x, [-1, 7 * 7 * conv6_num_filters])
        h_pool_flat_with_angle = tf.concat([h_pool_flat, angle], 1)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat_with_angle, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([num_hidden_fc1, num_hidden_fc2])
        b_fc2 = bias_variable([num_hidden_fc2])

        pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return pred, keep_prob


def main():
    # Load the data
    x_train, x_train_angle, y_train, x_test, x_test_angle, y_test = load_data()

    # Create the model
    x = tf.placeholder(tf.float32, [None, dimentions[0] * dimentions[1], num_images], name='x')
    x_angle = tf.placeholder(tf.float32, [None, 1], name='angle')
    y = tf.placeholder(tf.float32, [None, 2], name="y")

    conv_out = conv_tensor(x)

    pred, keep_prob = fully_connected(conv_out, x_angle)

    softmax_pred = tf.nn.softmax(pred, name='output')

    cross_validtion = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_validtion)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the global variables
    init = tf.global_variables_initializer()

    # Run the model
    with tf.Session() as sess:
        acc=0
        meanAcc_list=np.zeros(num_iter)
        epoch_list=[]
        maxI=0
        sess.run(init)
        for epoch in range(num_iter):
            print(epoch)
            for i in range(0, len(x_train), batch_size):
                x_batch, x_batch_angle = x_train[i:i + batch_size], x_train_angle[i:i+batch_size]
                if i + batch_size >= len(x_train):
                    x_batch, x_batch_angle = x_train[i:], x_train_angle[i:]

                y_batch = y_train[i:i + batch_size]

                # run a test on the dev set to get som feedback of how the model i doing
                train_accuracy = accuracy.eval(feed_dict={
                    x: x_batch, x_angle: x_batch_angle, y: y_batch, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
                acc+= train_accuracy
                maxI+=1

                train_step.run(feed_dict={x: x_batch, x_angle: x_batch_angle, y: y_batch, keep_prob: dropout_prob})
            epoch_list.append(epoch)
            meanAcc_list[epoch]=acc/maxI
            print('Mean training accuracy: ', meanAcc_list[epoch])
            maxI=0
            acc=0
        if dev_set_size != 0:
            print('test accuracy %g' % accuracy.eval(
                feed_dict={x: x_test, x_angle: x_test_angle, y: y_test, keep_prob: 1.0}))
        plt.plot(epoch_list, meanAcc_list, color='magenta', label='Mean training accuracy')
        plt.show()
        x_test = x_train = x_train_angle = x_test_angle = y_train = y_test = None

        # create the submission
        if make_csv:
            with open('fries.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['id','is_iceberg'])
                for i in (file_names):
                    x_test, x_test_angle, id = load_test_data(i)
                    for j in range(0, len(x_test), batch_size):
                        x_batch, x_batch_angle, id_batch = x_test[j:j + batch_size], x_test_angle[j:j + batch_size], id[j:j + batch_size]
                        if j + batch_size >= len(x_test):
                            x_batch, x_batch_angle, id_batch = x_test[j:], x_test_angle[j:], id[j:]

                        out = sess.run(softmax_pred, feed_dict={x: x_batch, x_angle: x_batch_angle, keep_prob: 1.0})
                        for k in range(len(out)):
                            writer.writerow([id_batch[k], out[k][0]])
                        x_batch = x_batch_angle = id_batch = None
                    x_test = x_test_angle = id = None
                    print("file_read:", i)


if __name__ == '__main__':
    main()
