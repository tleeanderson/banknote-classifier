from common import funcs
from sklearn import tree
import graphviz
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd

def graph(model, feats, plot_dir, name):
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=feats,
                                    class_names=['authentic', 'fraudulent'],
                                    filled=True, rounded=True, special_characters=True,
                                    leaves_parallel=True)
    graph = graphviz.Source(dot_data)
    out_path = os.path.join(plot_dir, name)
    graph.render(out_path)
    return out_path

def evaluate(model, model_params, xs, ys):
    search = GridSearchCV(estimator=model, param_grid=model_params, cv=5)
    search.fit(xs, ys)
    df = pd.DataFrame(search.cv_results_)
    return df[['param_ccp_alpha', 'rank_test_score', 'std_test_score', 'mean_test_score', 'params']]\
        .sort_values(by=['rank_test_score'])

def main(data_file, plot_dir):
    data = funcs.read_data(data_file)
    xs, ys = funcs.xs_and_ys(data)
    model = tree.DecisionTreeClassifier()
    param_space = {'ccp_alpha': np.linspace(0.001, 0.05, num=25)}
    cv_results = evaluate(model, param_space, xs, ys)

    print("Cross validation results: ")
    print(cv_results[list(cv_results)[:-1]])
    for i, hp in enumerate(cv_results['params']):
        out_model = tree.DecisionTreeClassifier(**hp)
        out_model.fit(xs, ys)
        op = graph(out_model, list(data)[:-1], plot_dir, "{}-tree".format(i))
    print("Wrote out {} trees to {} plot_dir".format(len(cv_results), plot_dir))
