import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Activation, Reshape, concatenate, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback


START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'
chars = set([START_CHAR, '\n', END_CHAR])
input_frame = 'shakespeare_short.txt'
model_fname = 'model_keras'
output_fname = 'output.txt'
batchout_fname = 'batch_out.txt'
USE_SIMPLE_MODEL = False


with open(input_frame) as f:
    for line in f:
        chars.update(list(line.strip().lower()))

char_indicies = {c: i for i, c in enumerate(sorted(list(chars)))}
char_indicies[PADDING_CHAR] = 0
indicies_to_chars = {i: c for c, i in char_indicies.items()}
num_chars = len(chars)

print(num_chars)


def get_one(i, sz):
    res = np.zeros(sz)
    res[i] = 1
    return res


char_vectors = {
    c: (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars)) for c, v in char_indicies.items()
}

sentence_end_markers = set('.!?')
sentences = []
current_sentence = ''

with open(input_frame, 'r') as f:
    for line in f:
        s = line.strip().lower()
        if len(s) > 0:
            current_sentence += s + '\n'
        if len(s) == 0 or s[-1] in sentence_end_markers:
            current_sentence = current_sentence.strip()
            if len(current_sentence) > 10:
                sentences.append(current_sentence)
            current_sentence = ''


def get_matrices(sentences, max_sentence_len):
    X = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), max_sentence_len, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(max_sentence_len + 1, PADDING_CHAR)
        for t in range(max_sentence_len):
            X[i, t, :] = char_vectors[char_seq[t]]
            y[i, t, :] = char_vectors[char_seq[t + 1]]
    return X, y


test_indicies = np.random.choice(range(len(sentences)), int(len(sentences) * 0.05))
sentences_train = [sentences[x] for x in set(range(len(sentences))) - set(test_indicies)]
sentences_test = [sentences[x] for x in test_indicies]

max_sentence_len = np.max([len(x) for x in sentences])

sentences_train = sorted(sentences_train, key=lambda x: len(x))
X_test, y_test = get_matrices(sentences_test, max_sentence_len)
batch_size = 16

print(sentences_train[1])
print(sentences_test[1])
print(X_test.shape)


def generate_batch():
    while True:
        for i in range(int(len(sentences_train) / batch_size)):
            sentences_batch = sentences_train[i * batch_size:(i + 1) * batch_size]
            yield get_matrices(sentences_batch, max_sentence_len)

class CharSampler(Callback):
    def __init__(self, char_vectors, model):
        self.char_vectors = char_vectors
        self.model = model

    def on_train_begin(self, logs={}):
        self.epoch = 0
        if os.path.isfile(output_fname):
            os.remove(output_fname)

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample_one(self, T):
        result = START_CHAR
        while len(result) < 500:
            Xsampled = np.zeros((1, len(result), num_chars))  # max_sentence_len
            for t, c in enumerate(list(result)):
                Xsampled[0, t, :] = self.char_vectors[c]
            ysampled = self.model.predict(Xsampled, batch_size=1)[0, :]
            yv = ysampled[len(result) - 1, :]
            selected_char = indicies_to_chars[self.sample(yv, T)]
            if selected_char == END_CHAR:
                break
            result = result + selected_char
        return result

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch + 1
        if self.epoch % 1 == 0:
            print('\nEpoch: %d text sampling:' % self.epoch)
            with open(output_fname, 'a') as outf:
                outf.write('\n========= Epoch %d =========' % self.epoch)
                for T in [.3, .5, .7, .9, 1.1]:
                    print('\tsampling, T= %.1f...' % T)
                    for _ in range(5):
                        self.model.reset_states()
                        res = self.sample_one(T)
                        outf.write('\nT=%.1f \n%s \n' % (T, res[1:]))

    def on_batch_end(self, batch, logs={}):
        if (batch + 1) % 10 == 0:
            print('\nBatch %d text sampling: ' % batch)
            with open(output_fname, 'a') as outf:
                outf.write('\n========= Batch %d =========' % batch)
                for T in [.3, .5, .7, .9, 1.1]:
                    print('\tsampling, T= %.1f...' % T)
                    for _ in range(5):
                        self.model.reset_states()
                        res = self.sample_one(T)
                        outf.write(res + '\n')


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        if (batch + 1) % 100 == 0:
            with open(batchout_fname, 'a') as outf:
                for i in range(100):
                    outf.write('%d\t%.6f\t%.6f\n' %
                               (batch + i - 99, self.loss[i - 100], self.acc[i - 100]))


if USE_SIMPLE_MODEL:
    # simple model
    vec = Input(shape=(None, num_chars))
    l1 = LSTM(128, activation='tanh', return_sequences=True)(vec)
    l1_d = Dropout(0.2)(l1)
    dense = TimeDistributed(Dense(num_chars))(l1_d)
    output_res = Activation('softmax')(dense)
    model = Model(input=vec, outputs=output_res)
else:
    # deep model
    vec = Input(shape=(None, num_chars))
    l1 = LSTM(128, activation='tanh', return_sequences=True)(vec)
    l1_d = Dropout(0.2)(l1)

    input2 = concatenate([vec, l1_d])
    l2 = LSTM(128, activation='tanh', return_sequences=True)(input2)
    l2_d = Dropout(0.2)(l2)

    input3 = concatenate([vec, l2_d])
    l3 = LSTM(128, activation='tanh', return_sequences=True)(input3)
    l3_d = Dropout(0.2)(l2)

    input_d = concatenate([l1_d, l2_d, l3_d])
    dense3 = TimeDistributed(Dense(num_chars))(input_d)
    output_res = Activation('softmax')(dense3)
    model = Model(input=vec, outputs=output_res)

model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.), metrics=['accuracy'])

cb_sampler = CharSampler(char_vectors, model)
cb_logger = CSVLogger(model_fname + '.log')
cb_checkpoint = ModelCheckpoint("model.hdf5", monitor='val_acc', save_best_only=True, save_weights_only=False)

model.fit_generator(generate_batch(),
                    int(len(sentences_train) / batch_size) * batch_size,
                    epochs=10,
                    verbose=True,
                    validation_data=(X_test, y_test),
                    callbacks=[cb_logger, cb_sampler, cb_checkpoint])
