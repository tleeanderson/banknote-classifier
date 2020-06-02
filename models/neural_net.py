import numpy as np
import pandas as pd
from scipy.stats import zscore
from matplotlib import pyplot
import functools
import operator as op
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

def standardize(df):
    for f in list(df)[:-1]:
        df.update(pd.DataFrame({f: zscore(df[f])}))

def split_data(df, sp=0.8):
    n = len(df.loc[df['class'] == 0]) / len(df)
    p = 1 - n
    train = int(len(df) * sp)
    trp = int(train * p)
    trn = train - trp
    
    pos = df.loc[df['class'] == 1].sample(frac=1)
    train_pos, test_pos = pos.iloc[:trp], pos.iloc[trp:]
    
    neg = df.loc[df['class'] == 0].sample(frac=1)
    train_neg, test_neg = neg.iloc[:trn], neg.iloc[trn:]

    return ((train_pos, train_neg), (test_pos, test_neg))

def plot_features(out_dir, data):
    xs = range(len(data))
    args = [(f, xs, data[f], c) for f, c \
            in zip(list(data), ('ro', 'bo', 'go', 'yo'))]
    for a in args:
        ar = a[1:]
        fig, ax = pyplot.subplots()
        ax.plot(*ar)
        fig.savefig(os.path.join(out_dir, "std-{}".format(a[0])))

def xs_and_ys(df):
    return np.asarray(df[list(df)[:-1]]), np.asarray(df[list(df)[-1]])

def train_net(data_file):
    data = pd.read_csv(data_file)
    data.drop_duplicates(keep='first', inplace=True, ignore_index=True, subset=list(data)[:-1])

    standardize(data)
    train, test = split_data(data)
    all_train = pd.concat(train).sample(frac=1)
    all_test = pd.concat(test).sample(frac=1)
    trn_x, trn_y = xs_and_ys(all_train)
    tst_x, tst_y = xs_and_ys(all_test)

    model = Sequential()
    model.add(Dense(3, input_dim=4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])

    model.fit(trn_x, trn_y, epochs=15, batch_size=5)
    return (model, tst_x, tst_y)

def roc_curve(tst_x, tst_y, model, plot_dir):
    loss, acc = model.evaluate(tst_x, tst_y)
    print("eval: loss: {:.04}, acc: {:.0%}".format(loss, acc))

    preds = model.predict(tst_x)
    ps, ys = zip(*[(preds[i][0], tst_y[i]) for i in range(len(preds))])
    fp_rate, tp_rate, th = metrics.roc_curve(np.array(ys), np.array(ps))
    sk_fig, sk_ax = pyplot.subplots()

    sk_ax.plot(fp_rate, tp_rate, 'b', [0, 1], [0, 1], '--r')
    sk_ax.set_xlabel('FPR')
    sk_ax.set_ylabel('TPR')
    auc = metrics.auc(fp_rate, tp_rate)
    sk_ax.legend(["AUC = {:.05}".format(auc)])
    sk_fig.savefig(os.path.join(plot_dir, 'roc_curve'))

