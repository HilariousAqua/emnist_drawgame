#imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LeakyReLU, AvgPool2D, BatchNormalization, Reshape
from keras.utils import Sequence, to_categorical
from keras.optimizers import Adam

import numpy as np
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn

#constants
IMG_ROWS = 28
IMG_COLS = 28
EPOCHS = 10

#functions
def flip_and_rotate(image):
    image = image.reshape(IMG_ROWS, IMG_COLS)
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# import from data
try:
    train_df = pd.read_csv("data/emnist-balanced-train.csv", header=None)
    test_df = pd.read_csv("data/emnist-balanced-test.csv", header=None)
    label_df = pd.read_csv("data/emnist-balanced-mapping.txt", delimiter=' ', index_col=0, header=None)
except:
    raise SystemExit("Files not found, try running getdata.py to generate.")

#process label dictionary
label_dictionary = {}

for index, label in enumerate(label_df[1]):
    label_dictionary[index] = chr(label)

#remove lowercase
train_df = train_df[train_df[0].isin(np.arange(0, 36))]

#reset index
train_df.reset_index(inplace=True)

#separate images and labels
x_train = train_df.loc[:, 1:] # images
y_train = train_df.loc[:, 0] # labels

#applying rotate to all images
x_train = np.apply_along_axis(flip_and_rotate, 1, x_train.values)

# scale values from 0 to 1 (float32)
x_train = x_train.astype('float32') / 255

# get num of classes
number_of_classes = y_train.nunique()

# one hot encoding
y_train = tf.keras.utils.to_categorical(y_train, number_of_classes)
y_train.shape

# reshape to 4d array
x_train = x_train.reshape(-1, IMG_ROWS, IMG_COLS, 1)

# model building

model = Sequential()
          
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', activation='relu', input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(MaxPooling2D((2, 2)))
          
model.add(Flatten())
          
model.add(Dense(128, activation='relu'))

model.add(Dense(number_of_classes, activation='softmax'))

# compiling model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# training
model.fit(x_train, y_train, epochs=EPOCHS, shuffle=True)

#testing with the csv

test_df = pd.read_csv('data/emnist-balanced-test.csv', header=None)
test_df = test_df[test_df[0].isin(np.arange(0, 36))]

x_test = test_df.loc[:, 1:]
y_test = test_df.loc[:, 0]

x_test = np.apply_along_axis(flip_and_rotate, 1, x_test.values)
y_test = to_categorical(y_test, number_of_classes)

x_test = x_test.astype('float32') / 255.0

x_test = x_test.reshape(-1, IMG_ROWS, IMG_COLS, 1)

# testing
test_acc = model.evaluate(x_test, y_test)
print(f"acc: {test_acc[1]}")

#export
if not os.path.exists("models"):
    os.mkdir("models")
model.save('models/emnist_model2.keras')



