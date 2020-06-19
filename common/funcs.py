import pandas as pd
from matplotlib import pyplot
import os
import numpy as np
from sklearn import metrics

def plot_features(out_dir, data):
    xs = range(len(data))
    args = [(f, xs, data[f], c) for f, c \
            in zip(list(data), ('ro', 'bo', 'go', 'yo'))]
    for a in args:
        ar = a[1:]
        fig, ax = pyplot.subplots()
        ax.plot(*ar)
        fig.savefig(os.path.join(out_dir, "std-{}".format(a[0])))

def roc_curve(ys_ps, plot_dir, model_name):
    rates = [(fpr, tpr, th, c) for (fpr, tpr, th), c \
             in [(metrics.roc_curve(np.array(ys), np.array(ps)), c) for ys, ps, c in ys_ps]]
    sk_fig, sk_ax = pyplot.subplots()
    for f, t, _, c in rates:
        sk_ax.plot(f, t, c)
    sk_ax.plot([0, 1], [0, 1], '--r')
    sk_ax.set_xlabel('FPR')
    sk_ax.set_ylabel('TPR')
    
    auc = [metrics.auc(fpr, tpr) for fpr, tpr, _, _ in rates]
    avg_auc, std_auc, mi, mx = np.average(auc), np.std(auc), np.min(auc), np.max(auc)
    sk_ax.legend(["AVG AUC = {:.03}\nSTD AUC = {:.03}\nMIN AUC = {:.03}\nMAX AUC = {:.03}"\
                  .format(avg_auc, std_auc, mi, mx)], loc='lower right')
    out_path = os.path.join(plot_dir, "{}-roc_curve".format(model_name))
    sk_fig.savefig(out_path)
    print("Saved roc curve to {}".format(out_path))    

def read_data(data_file):
    data = pd.read_csv(data_file)
    data.drop_duplicates(keep='first', inplace=True, ignore_index=True, subset=list(data)[:-1])
    return data

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

def xs_and_ys(df):
    return np.asarray(df[list(df)[:-1]]), np.asarray(df[list(df)[-1]])

def examples(train_split, test_split):
    all_train = pd.concat(train_split).sample(frac=1)
    all_test = pd.concat(test_split).sample(frac=1)
    trn_x, trn_y = xs_and_ys(all_train)
    tst_x, tst_y = xs_and_ys(all_test)
    return (trn_x, trn_y, tst_x, tst_y)

def sort_fmt_scores(scores):
    return ["{:0.5%}".format(s) for s in sorted(scores)]
