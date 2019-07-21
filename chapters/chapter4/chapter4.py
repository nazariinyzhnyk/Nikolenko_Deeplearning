from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
import plotly.graph_objects as go
from itertools import product
import pandas as pd


def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())],
                        columns=dictionary.keys())


def add_dense_layer(model, init, act, num_units=100, use_bn=False):
    if use_bn:
        model.add(BatchNormalization)
    model.add(Dense(num_units, kernel_initializer=init, activation=act))
    return model


def plot_acc_loss(names, values, show_fig=True, save_fig=True, fig_name='losses'):
    x_axis = [i + 1 for i in range(len(values[1]))]
    fig = go.Figure()
    for i in range(len(names)):
        fig.add_trace(go.Scatter(x=x_axis, y=values[i],
                                 mode='lines',
                                 name=names[i]))
    fig.update_layout(title=go.layout.Title(text=fig_name))
    if show_fig:
        fig.show()
    if save_fig:
        fig.write_image("images/" + fig_name + ".png")


def create_model(init, act='tanh', num_dense=3, num_units=100, use_bn=False):
    model = Sequential()
    model.add(Dense(num_units, input_shape=(28 * 28,), kernel_initializer=init, activation=act))

    for i in range(num_dense):
        add_dense_layer(model, init=init, act=act, num_units=num_units, use_bn=use_bn)

    model.add(Dense(10, kernel_initializer=init, activation='softmax'))
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

dictionary = {'initialization': ['uniform', 'glorot_normal', 'he_normal'],
              'optimizer': ['sgd', 'adam'],
              'use_bn': [True, False]}

grid = expand_grid(dictionary)
print(grid)
print(len(grid))

model_param_names = []
model_param_accuracies = []
model_param_losses = []

for i in range(len(grid)):
    model_param_names.append(grid['initialization'][i] + '_' + grid['optimizer'][i] + '_' + str(grid['use_bn'][i]))
    model = create_model('he_normal', act='tanh', num_dense=3, num_units=100, use_bn=False)  # uniform
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train, Y_train,
                     batch_size=64,
                     epochs=20,
                     verbose=1,
                     validation_data=(X_test, Y_test))
    model_param_accuracies.append(hist.history['val_acc'])
    model_param_losses.append(hist.history['val_loss'])

plot_acc_loss(model_param_names, model_param_losses, show_fig=True, save_fig=True, fig_name='losses')
plot_acc_loss(model_param_names, model_param_accuracies, show_fig=True, save_fig=True, fig_name='accuracies')
