"""
code to generate graph embeddings for income and occupation prediction
"""

import argparse
from bipartite_graph import BipartiteGraph

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate graph embeddings for income and occupation prediction',
                                     epilog='Embeddings are learned from Twitter social network data')
    parser.add_argument(
        'x_path', type=str,
        nargs='+', default='../local_resources/X_thresh10.p', help='the location of the graph')
    # parser.add_argument(
    #     'y_path', type=str,
    #     nargs='+', default='../local_resources/y_thresh10.p', help='the location of the labels')
    # parser.add_argument(
    #     '-nfolds', type=int,
    #     nargs='+', default=3, help='number of stratified folds to split the data into for cross-validation')
    # args = parser.parse_args()
    #
    # x, y = utils.read_data(args.x_path[0], args.y_path[0], threshold=0)
    # n_classes = len(np.unique(y))
    # print('n_classes: {}'.format(n_classes))
    # clf = AgeDetector(n_classes)
    # n_folds = args.nfolds[0]
    # y_pred = run_cv_pred(x, y, clf, n_folds)
