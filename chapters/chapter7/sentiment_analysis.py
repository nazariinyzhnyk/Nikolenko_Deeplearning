import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
top_words = 5000
max_review_length = 500
embedding_vector_length = 32
use_conv = True

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
np.load = np_load_old

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
if use_conv:
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
else:
    model.add(Dropout(0.2))
model.add(LSTM(100))
if not use_conv:
    model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: %.4f' % scores[1])
