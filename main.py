import argparse
import itertools
import matplotlib.pyplot as pyplot
import os
from mpl_toolkits import mplot3d
from common import funcs

def one_feature_plots(pos, neg, feats, plot_dir):
    for f in feats:
        pos_fs, neg_fs = list(pos[f]), list(neg[f])
        p_xs, n_xs = range(len(pos_fs)), range(len(neg_fs))
        fig, ax = pyplot.subplots()
        ax.plot(p_xs, pos_fs, 'ro', n_xs, neg_fs, 'bo')
        fig.savefig(os.path.join(plot_dir, "one-feat-{}".format(f)))

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

def parse_args():
    parser = argparse.ArgumentParser(description="Classify a banknote as fradulent or authentic")
    parser.add_argument("-i", "--input-file", required=False, 
                        help="UCI banknote CSV file. Defaults to dataset/data_banknote_authentication.csv", 
                        default='dataset/data_banknote_authentication.csv')
    parser.add_argument("-tn", "--train-neural-net", required=False, action='store_true',
                        help="Train the neural network. This will output a roc curve under" 
                        + " the plots directory.")
    parser.add_argument("-dt", "--train-decision-tree", required=False, action='store_true')
    parser.add_argument("-gp", "--generate-plots", required=False, action='store_true', 
                        help="Generate the feature plots. These will be written underneath" 
                        + " the plots directory.")
    parser.add_argument("-pd", "--plot-dir", required=False, default="./plots", 
                        help="Directory to write the plots. Defaults to $PROJECT_ROOT/plots.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    setup_out_dirs(args)

    if args.generate_plots:
        df = funcs.read_data(args.input_file)
        feats = list(df)[:-1]
        neg, pos = [df.loc[df['class'] == arg] for arg in (0, 1)]
        one_feature_plots(pos, neg, feats, args.plot_dir)
        two_feature_plots(pos, neg, feats, args.plot_dir)
        three_feature_plots(pos, neg, feats, args.plot_dir)
        print("Generated plots in directory {}".format(args.plot_dir))

    if args.train_neural_net:
        from models import neural_net as nn
        nn.main(args.input_file, args.plot_dir)

    if args.train_decision_tree:
        from models import decision_tree as dec_tree
        dec_tree.main(args.input_file, args.plot_dir)
