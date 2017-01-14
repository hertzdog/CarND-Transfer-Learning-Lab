import pickle
import numpy as np
import math
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.datasets import cifar10
from sklearn.utils import shuffle
from sklearn import preprocessing

# Normalize the data features to the variable X_normalized
def normalize(image_data):
    #a = 0.1
    #b = 0.9
    #channel_min = 0
    channel_max = 255
    #return a + ( ( (image_data - channel_min)*(b - a) )/( channel_max - channel_min ) )
    # NORMALIZE (from 0-255 to 0-1)
    image_data = image_data / channel_max
    # ZERO MEAN - CENTER (shift left 0,5)
    image_data = image_data - 0.5
    return image_data



# TODO: One Hot encode the labels to the variable y_one_hot




flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    # load raw data
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    # shuffle
    X_train, y_train = shuffle(X_train, y_train)

    X_normalized = normalize(X_train)

    lb = preprocessing.LabelBinarizer()
    y_one_hot = lb.fit_transform(y_train)

    dropout=0.5

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    model = Sequential()

    # 1st Layer - Convolutional with 32 filters, a 3x3 kernel, and valid padding before the flatten layer
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(32,32,3)))

    # 2nd Layer - Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))

    # 3rd Layer - Dropout
    model.add(Dropout(dropout))

    # 4th layer - ReLU activation layer
    model.add(Activation('relu'))

    # 5th Layer - Add a flatten layer
    model.add(Flatten(input_shape=(32, 32, 3)))

    # 6th Layer - Add a fully connected layer
    model.add(Dense(128))

    # 7th Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    # 8th Layer - 43 classes - 43 neurons - Output or 10 for cifar10
    model.add(Dense(10))

    # 9th layer softmax
    model.add(Activation('softmax'))


    # TODO: train your model here
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    phistory=model.fit(X_normalized, y_one_hot, batch_size=32, nb_epoch=10, \
          verbose=1, validation_split=0.2)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
