import csv
import argparse
import itertools
import matplotlib.pyplot as pyplot
import os

VARIANCE, SKEWNESS, CURTOSIS, ENTROPY = range(4)
FEAT_NAMES = ['variance', 'skewness', 'curtosis', 'entropy']

def read_data(file_path):
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=',')            
        return [(k, list(g)) for k, g in itertools.groupby(
            sorted([[float(n) for n in r[:-1]] + [int(r[-1])] for r in reader], 
                   key=lambda t: t[-1]), key=lambda t: t[-1])]

def plot_feature(feat, positives, negatives, plot_name, plot_dir, feat_names=FEAT_NAMES):
    prep = lambda data, f: [ex[f] for ex in data]
    pos, neg = [prep(data, feat) for data in (positives, negatives)]

    num_ex = min(len(pos), len(neg))
    xs = range(num_ex)
    pyplot.plot(xs, pos[:num_ex], 'ro', xs, neg[:num_ex], 'bo')
    pyplot.savefig(os.path.join(plot_dir, plot_name))

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
    
    parser.add_argument("-pd", "--plot-dir", required=False, default="plots")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    setup_out_dirs(args)

    neg_tup, pos_tup = sorted(read_data(args.input_file))
    nl, neg_ex = neg_tup
    pl, pos_ex = pos_tup

    print_stats(pos_ex, neg_ex)

    for f, pn in enumerate(FEAT_NAMES):
        plot_feature(f, pos_ex, neg_ex, pn, args.plot_dir)
