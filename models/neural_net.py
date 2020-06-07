import numpy as np
import pandas as pd
from scipy.stats import zscore
from matplotlib import pyplot
import functools
import operator as op
import os
from keras.models import Sequential
from keras.layers import Dense
from common import funcs

def standardize(df):
    for f in list(df)[:-1]:
        df.update(pd.DataFrame({f: zscore(df[f])}))

def model():
    model = Sequential()
    model.add(Dense(3, input_dim=4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model

def evaluate(tst_x, tst_y, model, plot_dir):
    loss, acc = model.evaluate(tst_x, tst_y)
    print("eval: loss: {:.04}, acc: {:.0%}".format(loss, acc))

    preds = model.predict(tst_x)
    ps, ys = zip(*[(preds[i][0], tst_y[i]) for i in range(len(preds))])

    funcs.roc_curve(ys, ps, plot_dir, "neural_net")

def main(data_file, plot_dir):
    data = funcs.read_data(data_file)
    standardize(data)
    train, test = funcs.split_data(data)
    trn_x, trn_y, tst_x, tst_y = funcs.examples(train, test)

    md = model()
    md.fit(trn_x, trn_y, epochs=15, batch_size=5)
    evaluate(tst_x, tst_y, md, plot_dir)
