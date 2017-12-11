# -*- coding: utf-8 -*-
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from numpy import genfromtxt

data2D = genfromtxt('matrix.csv', delimiter=',')
print("data loaded")
numObservations = (int)(data2D.size / 64)
data = data2D.reshape(numObservations, 8, 8)
print("data reshaped")
X = np.zeros((numObservations, 8, 8, 6))
y = genfromtxt('results.csv', delimiter=',')
for i in range(numObservations):
    if i % 10000 == 0:
        print(i)
    X[i, :, :, 0] = data[i, :, :] == 1 * np.ones((8, 8))
    X[i, :, :, 1] = data[i, :, :] == -1 * np.ones((8, 8))
    X[i, :, :, 2] = data[i, :, :] == 2 * np.ones((8, 8))
    X[i, :, :, 3] = data[i, :, :] == -2 * np.ones((8, 8))
    for a in range(8):
        for b in range(8):
            if a == 0 or a == 7 or b == 0 or b == 7:
                X[i, a, b, 4] = 1
            if a == 0 or a == 7:
                X[i, a, b, 5] = 1

trainMask = np.zeros(X.shape[0])
for i in range(len(trainMask)):
    if np.random.random() > 0.2:
        trainMask[i] = 1

Xtrain = X[trainMask == 1]
Xtest = X[trainMask == 0]

Ytrain = y[trainMask == 1]
Ytest = y[trainMask == 0]

model = Sequential()

model.add(Conv2D(20, (3, 3), input_shape=(8, 8, 6), padding='same', activation='relu'))
model.add(Conv2D(20, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Conv2D(20, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Conv2D(20, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Conv2D(20, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Conv2D(20, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.00001)))

model.add(Flatten())

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.00001)))
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])
for i in range(5):
    model.fit(Xtrain, Ytrain, batch_size=128, epochs=10, verbose=1)

    errorTrain = np.average(np.abs(model.predict(Xtrain)[:, 0] - Ytrain))
    errorTest = np.average(np.abs(model.predict(Xtest)[:, 0] - Ytest))
    print(errorTrain)
    print(errorTest)

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
