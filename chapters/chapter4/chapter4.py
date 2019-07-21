from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils


def create_model(init):
    model = Sequential()
    model.add(Dense(100, input_shape=(28*28, ), init=init, activation='tanh'))
    model.add(Dense(100, init=init, activation='tanh'))
    model.add(Dense(100, init=init, activation='tanh'))
    model.add(Dense(100, init=init, activation='tanh'))
    model.add(Dense(10, init=init, activation='softmax'))
    return model


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('\nShapes before transform:')
print('Train set shape : {}'.format(X_train.shape))
print('Train target len: {}'.format(len(y_train)))
print('Test set shape  : {}'.format(X_test.shape))
print('Test target len : {}'.format(len(y_test)))

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

X_train = X_train.reshape([-1, 28 * 28]) / 255
X_test = X_test.reshape([-1, 28 * 28]) / 255

print('\nShapes after transform:')
print('Train set shape   : {}'.format(X_train.shape))
print('Train target shape: {}'.format(Y_train.shape))
print('Test set shape    : {}'.format(X_test.shape))
print('Test target shape : {}'.format(Y_test.shape))

uniform_model = create_model('glorot_normal')  # uniform
uniform_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
uniform_model.fit(X_train, Y_train,
                  batch_size=64,
                  epochs=20,
                  verbose=1,
                  validation_data=(X_test, Y_test))

"""
note that seeds are not fixed and results may differ!

uniform distribution initialization:
train loss: .1542
train accuracy: .9653
val loss: .1670
val accuracy: .9535

glorot_normal (Xavier) distribution initialization:
train loss: .088
train accuracy: .9748
val loss: .1034
val accuracy: .9675

"""

