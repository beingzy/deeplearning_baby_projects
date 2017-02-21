""" Train deep-neural network to classify news with
    pre-trained word embedding.

    The pre-trained word embedding will be loaded as a frozen layer,
    whose weights would not be updated during training process.

    [tutorial](https://blog.keras.io/index.html)
    [official github file](https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py)
    [GloVe](http://nlp.stanford.edu/projects/glove/)

    Author: Yi Zhang <beingzy@gmail.com>
    Date: 2017/Feb/20
"""
from __future__ import print_function
import os 
import warnings
import numpy as np 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.utils.np_utils import to_categorical 
from keras.layers import Dense, Input, Flatten 
from keras.layers import Conv1D, MaxPooling1D, Embedding 
from keras.models import Model 
import sys

np.random.seed(42)


## =========================
## environment setting
## =========================
BASE_DIR = ''
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'data', '20_newsgroup')
GLOVE_DIR = os.path.join(BASE_DIR, 'data')

MAX_NB_WORDS = 20000 # max number of 
MAX_SEQUENCE_LENGTH = 1000 #  

VALIDATION_SPLIT = 0.2 # validation fraction

EMBEDDING_DIM = 100# embedding dimensions

# validating the runtime environment
if not os.path.exists(TEXT_DATA_DIR):
    msg = "Could not find text data for training from {}".format(
        TEXT_DATA_DIR) 
    warnings.warn(msg)

## =========================
## Preparing the text data
## =========================
print("Indexing word vectors.")
texts  = [] # list of text samples 
labels_index = {} # dictionary mapping label name to numeric id 
labels = [] # list of label ids 

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)

    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id 
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)

                with open(fpath, 'rb') as f:
                    texts.append(f.read())

                labels.append(label_id)

# convert bytes to string 
texts = [text.decode('ISO-8859-2') for text in texts]
print("Found {} texts.".format(len(texts)))

# ==================
# data preprocessing
# ==================
# preprocessing labelled data and feed into tensor 
tokenizer = Tokenizer(nb_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))

data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

# print out the dimension info for verification
n, m = data.shape
print('Shape of data tensor: {}, {}'.format(n, m))
n, m = labels.shape 
print('Shape of label tensor: {}, {}'.format(n, m))

# split the data into a training set and a validation set 
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = data[-nb_validation_samples:]

# ================
# model definition
# ================
# loading pre-train word embeddings
# url: http://nlp.stanford.edu/projects/glove/
print("loading pre-trained word embeddings...")
embedding_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f: 
    values = line.split() 
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs 

f.close() 
print('Found {:,} word vectors.'.format(len(embedding_index)))

# compute embedding matrix with word_index and pre-trained word embedding
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, ii in word_index.items():
    if ii >= nb_words: #MAX_NB_WORDS:
        continue

    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: 
        # words not found in embedding index wil be all-zeros
        embedding_matrix[ii] = embedding_vector

# create a frozen layer to utilize embedding matrix 
# note trainable = False
embedding_layer = Embedding(nb_words, 
                            EMBEDDING_DIM, 
                            weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print("The forzen layer of pre-trainned embedding had been created.")

# traing 1D convnet 
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x) # global max pooling 
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['acc'])

# training model, exciting !!!!
model.fit(x_train, y_train, validation_data=(x_val, y_val), 
          nb_epoch=2)
