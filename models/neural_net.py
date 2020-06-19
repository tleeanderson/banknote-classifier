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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import ParameterGrid

def standardize(df):
    for f in list(df)[:-1]:
        df.update(pd.DataFrame({f: zscore(df[f])}))

def simple_model(kernel_regularizer):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model

def complex_model(kernel_regularizer):
    model = Sequential()
    model.add(Dense(3, input_dim=4, activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    return model

def evaluate(model_params, xs, ys, tr_params, bm_params):
    st_kf = StratifiedKFold(n_splits=5)
    row_keys = ['train_loss', 'train_accuracy', 'test_loss', 'test_accuracy', 
                'train_params', 'model_params', 'model_name']
    cv_results = {k: [] for k in row_keys}
    ys_ps = {mod_func.__name__: [] for mod_func, _ in model_params}
    for mod_func, params in model_params:
        for ps in list(ParameterGrid(params)):
            for tr_ind, tst_ind in st_kf.split(xs, ys):
                tr_x, tr_y = xs[tr_ind], ys[tr_ind]
                tst_x, tst_y = xs[tst_ind], ys[tst_ind]
                bmps = {k: ps[k] for k in bm_params}
                tps = {k: ps[k] for k in tr_params}
                train_model = mod_func(**bmps)
                th = train_model.fit(tr_x, tr_y, **tps)
                tr_l, tr_a = th.history['loss'][-1], th.history['accuracy'][-1]
                tst_l, tst_a = train_model.evaluate(tst_x, tst_y, verbose=0)

                cv_results[row_keys[0]].append(tr_l)
                cv_results[row_keys[1]].append(tr_a)
                cv_results[row_keys[2]].append(tst_l)
                cv_results[row_keys[3]].append(tst_a)
                cv_results[row_keys[4]].append(tps)
                cv_results[row_keys[5]].append(bmps)
                cv_results[row_keys[6]].append(mod_func.__name__)

                ys_ps[mod_func.__name__].append((tst_y, train_model.predict(tst_x)))

    return (pd.DataFrame(cv_results), ys_ps)

def main(data_file, plot_dir):
    data = funcs.read_data(data_file)
    standardize(data)
    xs, ys = funcs.xs_and_ys(data)

    model_params = [(simple_model, {'epochs': [1], 'batch_size': [5], 
                                    'verbose': [0], 'kernel_regularizer': ['l1', 'l2']}),
                    (complex_model, {'epochs': [1], 'batch_size': [5], 
                                     'verbose': [0], 'kernel_regularizer': ['l1', 'l2']})]
    train_params = {'epochs', 'verbose', 'batch_size'}
    build_model_params = {'kernel_regularizer'}
    cv_data, ys_ps = evaluate(model_params, xs, ys, train_params, build_model_params)

    print("Cross validation results:\n", cv_data)

    for mn, ps in ys_ps.items():
        funcs.roc_curve(ps, plot_dir, mn)
