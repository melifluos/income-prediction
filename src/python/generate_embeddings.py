"""
code to generate graph embeddings for income and occupation prediction
"""

import argparse
from bipartite_graph import BipartiteGraph
import utils
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate graph embeddings for income and occupation prediction',
                                     epilog='Embeddings are learned from Twitter social network data')
    parser.add_argument(
        'x_path', type=str,
        nargs=1, default='../local_resources/X_thresh10.p', help='the location of the graph')

    parser.add_argument(
        'out_path', type=str,
        nargs=1, default='../local_resources/X_thresh10.p', help='the location to write the embeddings')

    parser.add_argument(
        'walk_path', type=str,
        nargs='?', default="",
        help='the location of the random walk data. The default is to generate new random walks in local_resources')

    parser.add_argument(
        '--size', type=int,
        nargs='?', default=32, help='the embedding size')

    parser.add_argument(
        '--num_walks', type=int,
        nargs='?', default=10, help='the number of random walks to originate from each vertex')

    parser.add_argument(
        '--walk_length', type=int,
        nargs='?', default=80, help='the length of each random walk')

    args = parser.parse_args()
    size = args.size
    print 'learning embeddings of dimension {}'.format(args.size)
    x = utils.read_pickle(args.x_path[0])
    g = BipartiteGraph(x)
    print 'walk path: {}'.format(args.walk_path)
    print 'x path: {}'.format(args.x_path)
    if args.walk_path == "":
        print 'generating new random walk dataset'
        print 'building edges'
        g.build_edge_array()
        print 'generating walks'
        walks = g.generate_walks(args.num_walks, args.walk_length)
        df = pd.DataFrame(walks)
        walk_path = 'local_resources/walks_thresh10_num_{}_length_{}'.format(args.num_walks, args.walk_length)
        df.to_csv(walk_path, index=False, header=None)
    else:
        print 'learning embeddings'
        walks = pd.read_csv(args.walk_path,
                            header=None).values
    g.learn_embeddings(walks, size, args.out_path)
