from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from random import shuffle
import numpy as np
import os
import json

# Trying out Keras. This is a pretty straightforward conv net

batch_size = 300
num_classes = 1
epochs = 300
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'trained_model.h5'

cv_size = 160

with open('train_processed_cropped.json') as f:
    print('reading', f)
    dataset = json.load(f)
    print('dataset size', len(dataset))

shuffle(dataset)

x_train = np.empty((len(dataset) - cv_size, 50, 50, 3))
y_train = np.empty((len(dataset) - cv_size))
x_test = np.empty((cv_size, 50, 50, 3))
y_test = np.empty((cv_size))

i = 0
for item in dataset[0:-cv_size]:
    band1 = np.asarray(item['band_1']).reshape((50, 50))
    band2 = np.asarray(item['band_2']).reshape((50, 50))
    band3 = np.asarray(item['band_nabla']).reshape((50, 50))
    x_train[i] = np.dstack((band1, band2, band3))
    y_train[i] = item['is_iceberg'] == 1  # np.array((item['is_iceberg'] == 1, item['is_iceberg'] == 0))
    i += 1

i = 0
for item in dataset[len(dataset) - cv_size:]:
    band1 = np.asarray(item['band_1']).reshape((50, 50))
    band2 = np.asarray(item['band_2']).reshape((50, 50))
    band3 = np.asarray(item['band_nabla']).reshape((50, 50))
    x_test[i] = np.dstack((band1, band2, band3))
    y_test[i] = item['is_iceberg'] == 1  # np.array((item['is_iceberg'] == 1, item['is_iceberg'] == 0))
    i += 1

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(y_train)
print(np.sum(y_test))

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

print(model.predict(x_test))

# Save model and weights
model.save(model_name)
print('Saved trained model at %s ' % model_name)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
