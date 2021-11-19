import pandas as pd
import jieba.posseg as pseg
import os
import tensorflow as tf
import torch.nn as nn
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Input
from keras.layers import Embedding, LSTM, concatenate, Dense
from keras.models import Model

tf.compat.v1.disable_eager_execution()

TRAIN_DATA = "data/train.csv"
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 30
VALIDATION_RATIO = 0.1
RANDOM_STATE = 9527
NUM_EMBEDDING_DIM = 256
NUM_LSTM_UNITS = 128
NUM_CLASSES = 3
BATCH_SIZE = 512
EPOCHS = 100


def build_siamese_network():
    top_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    bm_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')

    embedding_layer = Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM)

    top_embedded = embedding_layer(top_input)
    bm_embedded =embedding_layer(bm_input)

    shared_lstm = LSTM(NUM_LSTM_UNITS)
    top_output = shared_lstm(top_embedded)
    bm_output = shared_lstm(bm_embedded)

    merged = concatenate([top_output, bm_output], axis=-1)
    dense = Dense(units=NUM_CLASSES, activation='softmax')

    predictions = dense(merged)

    model = Model(inputs=[top_input, bm_input], outputs=predictions)

    return model

def jieba_tokenizer(text):

    words = pseg.cut(text)
    return ' '.join([
        word for word, flag in words if flag != 'x'])

def main():

    train_df = pd.read_csv(TRAIN_DATA)

    cols = ['title1_zh',
        'title2_zh',
        'label']

    train_df = train_df.loc[:, cols]

    train_df["title1_tokenized"] = train_df.loc[:, "title1_zh"].astype(str).apply(jieba_tokenizer)

    train_df["title2_tokenized"] = train_df.loc[:, "title2_zh"].astype(str).apply(jieba_tokenizer)

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)

    corpus_x1 = train_df.title1_tokenized
    corpus_x2 = train_df.title2_tokenized
    corpus = pd.concat([corpus_x1, corpus_x2])

    tokenizer.fit_on_texts(corpus)


    x1_train = tokenizer.texts_to_sequences(corpus_x1)
    x2_train = tokenizer.texts_to_sequences(corpus_x2)

    x1_train = keras.preprocessing.sequence.pad_sequences(x1_train, maxlen=MAX_SEQUENCE_LENGTH)
    x2_train = keras.preprocessing.sequence.pad_sequences(x2_train, maxlen=MAX_SEQUENCE_LENGTH)

    label_to_index = {
        'unrelated': 0,
        'agreed': 1,
        'disagreed': 2
    }

    y_train = train_df.label.apply(lambda x: label_to_index[x])
    y_train = np.asarray(y_train).astype('float32')
    y_train = keras.utils.np_utils.to_categorical(y_train)

    x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, test_size=VALIDATION_RATIO, random_state=RANDOM_STATE)

    model = build_siamese_network()
    model.summary()
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x = [x1_train, x2_train],
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([x1_val, x2_val], y_val),
        shuffle=True
    )


    model.save('model/my_model')

if __name__ == "__main__":
    main()

