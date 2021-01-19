import strct
#"python.jediEnabled": true,

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow.python.keras as keras   # issue_1: unresolved import
from tensorflow.python.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.linear_model import RidgeClassifier, LogisticRegression


# version of Python - 32 or 64
print(f"python bit: {struct.calcsize('P') * 8}")

# XOR data 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float')
y = np.array([0, 1, 1, 0], dtype='float')

# ml
'''
model_LS = RidgeClassifier(alpha=0)
model_LS.fit(X, y)
print(f"LS accuracy: {model_LS.score(X, y)}")

model_LR = LogisticRegression(penalty='none')
model_LR.fit(X, y)
model_LS.score(X, y)
print(f"LR accuracy: {model_LR.score(X, y)}")
'''

# dl
'''
model_NN = keras.models.Sequential()
model_NN.add(keras.layers.Dense(10, activation='relu'))
model_NN.add(keras.layers.Dense(1, activation='sigmoid'))
model_NN.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
model_NN.fit(X, y, epochs=10000, verbose=0)
print(f"prediction: {model_NN.predict(X)} \n"\
      f"evaluate: {model_NN.evaluate(X, y)}")
'''

# MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

'''
plt.figure()
plt.imshow(X_train[0], cmap='gray')
plt.colorbar()
plt.xlabel(y_train[0], fontsize=18)
plt.show()
plt.savefig('mnist.png')

model_nn = keras.models.Sequential()
model_nn.add(Flatten(input_shape=(28, 28)))
model_nn.add(Dense(128, activation='relu'))
model_nn.add(Dense(10, activation='softmax'))
model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                 metrics=['acc'])
model_nn.fit(X_train, y_train, epochs=5, verbose=0)
print(f"evaluation: {model_nn.evaluate(X_test, y_test)}")
'''

# from scratch

inputs = Input(shape=(28, 28))
flatten = Flatten()(inputs)
hidden = Dense(128, activation='relu')(flatten)
outputs = Dense(10, activation='softmax')(hidden)

model_f = Model(inputs=inputs, outputs=outputs)
model_f.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                metrics=['acc'])
model_f.fit(X_train, y_train, epochs=10, verbose=0)
print(f"evaluation: {model_f.evaluate(X_test, y_test)}")
