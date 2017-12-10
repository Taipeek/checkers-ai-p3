# -*- coding: utf-8 -*-

import numpy as np
from keras import regularizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from numpy import genfromtxt
import theano

# print(theano.config)


data2D = genfromtxt('matrix_10000.csv', delimiter=',').reshape(10000, 8, 8)
numObservations = (int)(data2D.size / 64)
data = data2D.reshape(numObservations, 8, 8)
X = np.zeros((numObservations, 8, 8, 4))
y = genfromtxt('results_10000.csv', delimiter=',')
for i in range(numObservations):
    for j in range(4):
        if j == 0:
            X[i, :, :, j] = data[i, :, :] == 1 * np.ones((8, 8))
        elif j == 1:
            X[i, :, :, j] = data[i, :, :] == -1 * np.ones((8, 8))
        elif j == 2:
            X[i, :, :, j] = data[i, :, :] == 2 * np.ones((8, 8))
        elif j == 3:
            X[i, :, :, j] = data[i, :, :] == -2 * np.ones((8, 8))

trainMask = np.zeros(X.shape[0])
for i in range(len(trainMask)):
    if np.random.random() > 0.2:
        trainMask[i] = 1

Xtrain = X[trainMask == 1]
Xtest = X[trainMask == 0]

Ytrain = y[trainMask == 1]
Ytest = y[trainMask == 0]

model = Sequential()

model.add(Conv2D(8, (2, 2), input_shape=(8, 8, 4), padding='same', activation='relu'))
model.add(Conv2D(5, (4, 4), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(3, (6, 6), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Conv2D(1, (8, 8), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Flatten(input_shape=(8, 8, 4)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.001)))
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])
for i in range(1):
    model.fit(Xtrain, Ytrain, batch_size=64, epochs=10, verbose=1)

    errorTrain = np.average(np.abs(model.predict(Xtrain)[:, 0] - Ytrain))
    errorTest = np.average(np.abs(model.predict(Xtest)[:, 0] - Ytest))
    print(errorTrain)
    print((errorTest, '\n\n'))
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'

# returns a compiled model
# identical to the previous one

"""
model.add(Dropout(0.2))
model.add(Conv2D(3,(2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(3,(2, 2), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.000001)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.000001)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
for epoch in range(0,50):
    model.fit(x_train, y_train, batch_size=100, epochs=1, verbose=1)
    score=model.evaluate(x_test, y_test, verbose=0)
    print(score)
"""
