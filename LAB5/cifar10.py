"""
Authors:
    Adam Tomporowski, s16740
    Filip Bianga, s19329

Cifar10 Neural Network
-----------------------------------------------------------------------------------------------------
CIFAR is an acronym that stands for the Canadian Institute For Advanced Research and the
CIFAR-10 dataset was developed along with the CIFAR-100 dataset by researchers at the CIFAR institute.

CIFAR-10 is a well-understood dataset and widely used for benchmarking computer vision
algorithms in the field of machine learning.

Dataset: https://www.cs.toronto.edu/%7Ekriz/cifar.html
"""

# Based on: https://www.tensorflow.org/tutorials/images/cnn
# Based on: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# load dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
X = np.concatenate((X_train, X_test))
y = np.concatenate((Y_train, Y_test))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# normalize to range 0-1
X_train = X_train / 255
X_test = X_test / 255



def first_model_train():
    """
    -------------
    Prepare model
    -------------
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    """
    -----------------------
    Compile and train model
    -----------------------
    """
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=16)
    """
    ------------------------------------------------------
    Print loss and accuracy values for train and test sets
    ------------------------------------------------------
    First Output:
    TRAIN SETS
    loss: 37.29
    accuracy: 87.09
    ------------------
    TEST SETS
    loss: 110.93
    accuracy: 68.13
    """

    print('Cifar10 neural network - first model: ')
    loss, accuracy = model.evaluate(X_train, y_train)
    print('TRAIN SETS')
    print('loss: %.2f' % (loss * 100))
    print('accuracy: %.2f' % (accuracy * 100))
    print('----------------------------')
    loss_test, accuracy_test = model.evaluate(X_test, y_test)
    print('TEST SETS')
    print('loss:, %.2f' % (loss_test * 100))
    print('accuracy: %.2f' % (accuracy_test * 100))


def second_model_train():
    """
    -------------
    Prepare model
    -------------
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(48, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    """
    -----------------------
    Compile and train model
    -----------------------
    """
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=32)
    """
    ------------------------------------------------------
    Print loss and accuracy values for train and test sets
    ------------------------------------------------------
    First Output:
    TRAIN SETS
    loss: 31.32
    accuracy: 88.97
    ------------------
    TEST SETS
    loss: 119.22
    accuracy: 68.93
    """
    print('Cifar10 neural network - second model: ')
    loss, accuracy = model.evaluate(X_train, y_train)
    print('TRAIN SETS')
    print('loss: %.2f' % (loss * 100))
    print('accuracy: %.2f' % (accuracy * 100))
    print('----------------------------')
    loss_test, accuracy_test = model.evaluate(X_test, y_test)
    print('TEST SETS')
    print('loss:, %.2f' % (loss_test * 100))
    print('accuracy: %.2f' % (accuracy_test * 100))


if __name__ == "__main__":

    print("Cifar10 classification")
    choose = int(input("Choose the training model, 1 - standard or 2 - extended: "))

    #We choose which model train want
    if choose == 1:
        first_model_train()
    elif choose == 2:
        second_model_train()
