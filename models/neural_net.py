import numpy as np
import pandas as pd
from scipy.stats import zscore
from keras.models import Sequential
from keras.layers import Dense
from common import funcs
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

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
                'train_params', 'model_params', 'model_name', 'ys_ps', 'auc']
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'r']
    ci = 0
    cv_results = {k: [] for k in row_keys}
    ys_ps = {mod_func.__name__: [] for mod_func, _ in model_params}
    for mod_func, params in model_params:
        print("Training {}".format(mod_func.__name__))
        for ps in tqdm(list(ParameterGrid(params))):
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
                cv_results[row_keys[4]].append(tuple(tps.items()))
                cv_results[row_keys[5]].append(tuple(bmps.items()))
                cv_results[row_keys[6]].append(mod_func.__name__)
                ys_ps = (tst_y, train_model.predict(tst_x), colors[ci])
                cv_results[row_keys[7]].append(ys_ps)
                fpr, tpr, _ = metrics.roc_curve(np.array(ys_ps[0]), np.array(ys_ps[1]))
                cv_results[row_keys[8]].append(metrics.auc(fpr, tpr))

            ci = (ci + 1) % len(colors)

    return pd.DataFrame(cv_results)

def cv_results(cv_data):
    agg_funcs = ['mean', 'std', 'min', 'max']
    output = cv_data.groupby(['model_name', 'model_params', 'train_params'])\
                   .agg({'test_accuracy': agg_funcs,
                         'train_accuracy': agg_funcs,
                         'test_loss': agg_funcs,
                         'train_loss': agg_funcs,
                         'model_name': ['first'],
                         'model_params': ['first'],
                         'train_params': ['first'],
                         'ys_ps': ['sum'],
                         'auc': agg_funcs})
    return output

def roc_curves(cv_data, plot_dir):
    assoc_ys_ps = lambda ys_ps: [tuple(ys_ps[si:si+3]) for si in range(0, len(ys_ps), 3)]
    tup_to_str = lambda t: "_".join(["{}_{}".format(n, v) for n, v in t])
    for r in range(len(cv_data)):
        ys_ps = assoc_ys_ps(cv_data.iloc[r][('ys_ps', 'sum')])
        fn_params = "_".join([tup_to_str(cv_data.iloc[r][(col, 'first')]) \
                              for col in ('model_params', 'train_params')])
        fn = "{}_{}".format(cv_data.iloc[r][('model_name', 'first')], fn_params)
        funcs.roc_curve(ys_ps, plot_dir, fn)

def main(data_file, plot_dir):
    data = funcs.read_data(data_file)
    standardize(data)
    xs, ys = funcs.xs_and_ys(data)

    model_params = [(simple_model, {'epochs': [5, 10], 'batch_size': [3, 4], 
                                    'verbose': [0], 'kernel_regularizer': ['l1', 'l2']}),
                    (complex_model, {'epochs': [10, 15], 'batch_size': [3, 4], 
                                     'verbose': [0], 'kernel_regularizer': ['l1', 'l2']})]

    train_params = {'epochs', 'verbose', 'batch_size'}
    build_model_params = {'kernel_regularizer'}
    cv_data = evaluate(model_params, xs, ys, train_params, build_model_params)
    res = cv_results(cv_data)
    print("CV accuracy results:\n", res[[('test_accuracy', 'mean'), ('test_accuracy', 'std')]]\
          .sort_values(by=[('test_accuracy', 'mean')], ascending=False))

    print("\nCV auc results:\n", res[[('auc', 'mean'), ('auc', 'std')]]\
          .sort_values(by=[('auc', 'mean')], ascending=False), "\n")
    roc_curves(res, plot_dir)
