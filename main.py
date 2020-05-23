import csv
import argparse
import itertools
import matplotlib.pyplot as pyplot
import os
import classifier
import pandas as pd
from mpl_toolkits import mplot3d

def standard_plots(pos, neg, feats, plot_dir):
    for f in feats:
        pos_fs, neg_fs = list(pos[f]), list(neg[f])
        p_xs, n_xs = range(len(pos_fs)), range(len(neg_fs))
        fig, ax = pyplot.subplots()
        ax.plot(p_xs, pos_fs, 'ro', n_xs, neg_fs, 'bo')
        fig.savefig(os.path.join(plot_dir, "standard-{}".format(f)))

def two_feature_plots(pos, neg, feats, plot_dir):
    for f1, f2 in itertools.combinations(feats, 2):
        px, py = list(pos[f1]), list(pos[f2])
        nx, ny = list(neg[f1]), list(neg[f2])
        fig, ax = pyplot.subplots()
        ax.plot(px, py, 'ro', nx, ny, 'bo')
        fig.savefig(os.path.join(plot_dir, "two-feat-{}-{}".format(f1, f2)))

def three_feature_plots(pos, neg, feats, plot_dir):
    for f1, f2, f3 in itertools.combinations(feats, 3):
        pos_dp, neg_dp = [[list(df[f]) for f in (f1, f2, f3)] for df in (pos, neg)]
        fig = pyplot.figure()
        ax = pyplot.axes(projection='3d')
        for d, c, m in ((pos_dp, 'r', 'o'), (neg_dp, 'b', 'o')):
            ax.scatter3D(d[0], d[1], d[2], c=c, marker=m)
        fig.savefig(os.path.join(plot_dir, "three-feat-{}-{}-{}".format(f1, f2, f3)))

def setup_out_dirs(args):
    os.makedirs(args.plot_dir, exist_ok=True)

def print_stats(pos, neg):
    lp, ln = len(pos), len(neg)
    total = lp + ln
    pct = lambda t, ds: round(ds / t * 100, 2)
    print("neg (authentic): {}, pos (fradulent): {}, total: {}"\
          .format((ln, pct(total, ln)), (lp, pct(total, lp)), total))

def parse_args():
    parser = argparse.ArgumentParser(description="Classify a banknote as fradulent or authentic")
    parser.add_argument("-i", "--input-file", required=True)
    
    parser.add_argument("-gp", "--generate-plots", required=False, action='store_true')
    parser.add_argument("-pd", "--plot-dir", required=False, default="plots")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    setup_out_dirs(args)
    df = pd.read_csv(args.input_file)
    feats = list(df)[:-1]
    neg, pos = [df.loc[df['class'] == arg] for arg in (0, 1)]
    if args.generate_plots:
        standard_plots(pos, neg, feats, args.plot_dir)
        two_feature_plots(pos, neg, feats, args.plot_dir)
        three_feature_plots(pos, neg, feats, args.plot_dir)
