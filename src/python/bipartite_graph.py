"""
Efficient random walk generation for bipartite graphs
"""
import numpy as np
from numpy.random import randint
from datetime import datetime
import utils
import pandas as pd
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix

__author__ = 'benchamberlain'


class BipartiteGraph:
    """
    A binary graph
    """

    def __init__(self, adj=np.array([])):
        try:
            self.adj = adj.tocsr()
            self.row_deg = np.array(adj.sum(axis=1), dtype=int).squeeze()
            self.col_deg = np.array(adj.sum(axis=0), dtype=int).squeeze()
            self.n_rows = len(self.row_deg)
            self.n_cols = len(self.col_deg)
            self.row_edges = np.zeros(shape=(self.n_rows, max(self.row_deg)), dtype=np.uint32)
            self.col_edges = np.zeros(shape=(self.n_cols, max(self.col_deg)), dtype=np.uint32)
            print 'generated row_edges of shape {0} and column edges of shape {1}'.format(self.row_edges.shape,
                                                                                          self.col_edges.shape)
        except:
            print 'initialisation not complete, probably because no sparse matrix was passed to the constructor'

    def build_edge_array(self):
        """
        construct an array of edges. Instead of a binary representation each row contains the index of vertices reached
        by outgoing edges padded by zeros to the right
        0 0 1
        0 0 1
        1 1 0

        becomes

        2 0
        2 0
        0 1

        :return: None
        """
        self.build_row_array()
        self.build_col_array()

    def build_row_array(self):
        """
        Get the edges from the rows
        :return:
        """
        for row_idx in range(self.n_rows):
            # get the indices of the vertices that this vertex connects to
            z = self.adj[row_idx, :].nonzero()[1]
            # add these to the left hand side of the edge array
            self.row_edges[row_idx, 0:len(z)] = z

    def build_col_array(self):
        """
        calculate the edges from the columns
        :return:
        """
        csc_adj = self.adj.tocsc()
        for col_idx in range(self.n_cols):
            # get the indices of the vertices that this vertex connects to
            z = csc_adj[:, col_idx].nonzero()[0]
            # add these to the left hand side of the edge array
            self.col_edges[col_idx, 0:len(z)] = z

    def sample_next_vertices(self, current_vertices, degs):
        """
        get the next set of vertices for the random walks
        :return: next_vertices np.array shape = (len(current_vertices), 1)
        """
        current_degrees = degs[current_vertices]
        # sample an index into the edges array for each walk
        next_vertex_indices = np.array(map(lambda x: randint(x), current_degrees))
        return next_vertex_indices

    def initialise_walk_array(self, num_walks, walk_length):
        """
        Build an array to store the random walks with the initial starting positions in the first column. The order of
        the nodes is randomly shuffled as this is well known to speed up SGD convergence (Deepwalk: online learning of
        social representations)
        :return: A numpy array of shape = (n_vertices * num_walks, walk_length) which is all zero except for the first
        column
        """
        initial_vertices = np.arange(self.n_rows)
        # Add an extra column, which gets trimmed off later, but needed as the walk
        # is taking 2 steps at a time
        walks = np.zeros(shape=(self.n_rows * num_walks, walk_length + 1), dtype=np.uint32)
        walk_starts = np.tile(initial_vertices, num_walks)
        print 'shuffling walks to improve SGD conversion later'
        np.random.shuffle(walk_starts)
        walks[:, 0] = walk_starts
        print 'constructed random walk array of shape {0}'.format(walks.shape)
        return walks

    def biased_sample(self, p, deg):
        """
        either samples the previous node or a different node
        :param p: The repulsion factor - large p makes returning to the previous node less likely
        :param deg: the degree of the current node
        :return: either -1 for the previous vertex or an index
        """

    def generate_walks(self, num_walks, walk_length):
        """
        generate random walks
        :param num_walks the number of random walks per vertex
        :param walk_length the length of each walk
        :return: A numpy array of shape = (n_vertices * num_walks, walk_length)
        """
        assert walk_length % 2 == 0
        assert self.row_deg.min() > 0
        assert self.col_deg.min() > 0
        print 'initialising walks'
        walks = self.initialise_walk_array(num_walks, walk_length)

        for walk_idx in xrange(0, walk_length, 2):
            print 'generating walk step {}'.format(walk_idx)
            # get the vertices we're starting from
            current_vertices = walks[:, walk_idx]
            # get the indices of the next vertices. This is the random bit
            next_vertex_indices = self.sample_next_vertices(current_vertices, self.row_deg)
            next_vertices = self.row_edges[current_vertices, next_vertex_indices]
            # store distinct vertex indices for the columns by adding the max index for the rows
            walks[:, walk_idx + 1] = next_vertices + self.n_rows
            # get the indices of the next vertices. This is the random bit
            next_vertex_indices = self.sample_next_vertices(next_vertices, self.col_deg)
            walks[:, walk_idx + 2] = self.col_edges[next_vertices, next_vertex_indices]
        # little hack to make the right length walk
        return walks[:, :-1]

    def learn_embeddings(self, walks, size, outpath, window_size=10):
        """
        learn a word2vec embedding using the gensim library.
        :param walks: An array of random walks of shape (num_walks, walk_length)
        :param size: The number of dimensions in the embedding
        :param outpath: Path to write the embedding
        :returns None
        """
        # gensim needs an object that can iterate over lists of unicode strings. Not ideal for this application really.
        walk_str = walks.astype(str)
        walk_list = walk_str.tolist()

        model = Word2Vec(walk_list, size=size, window=window_size, min_count=0, sg=1, workers=4,
                         iter=5)
        model.wv.save_word2vec_format(outpath)


def scenario_debug():
    x = csr_matrix(np.array([[0, 1, 0, 1, 0],
                             [0, 0, 1, 1, 0],
                             [0, 1, 0, 1, 1],
                             [1, 1, 1, 0, 0],
                             [0, 0, 0, 1, 1]]))
    print 'reading data'
    s = datetime.now()
    # x = csr_matrix(np.array([[0, 1], [1, 0]]))
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(1, 4)
    print datetime.now() - s, ' s'
    print walks.shape


def scenario_build_small_age_embedding():
    print 'reading data'
    x, y = utils.read_data('resources/test/X.p', 'resources/test/y.p', 1)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    g.learn_embeddings(walks, 128, 'resources/test/test128.emd')
    print datetime.now() - s, ' s'
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv('resources/test/walks.csv', index=False, header=None)


def scenario_build_large_age_embedding():
    print 'reading data'
    x, y = utils.read_data('resources/test/X_large.p', 'resources/test/y_large.p', 1)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    g.learn_embeddings(walks, 128, 'resources/test/test128_large.emd')
    print datetime.now() - s, ' s'
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv('resources/test/walks_large.csv', index=False, header=None)


def scenario_build_balanced6_embeddings():
    print 'reading data'
    x, y = utils.read_data('resources/test/balanced6X.p', 'resources/test/balanced6y.p', 1)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    g.learn_embeddings(walks, 128, 'resources/test/balanced6.emd')
    print datetime.now() - s, ' s'
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv('resources/test/balanced6_walks.csv', index=False, header=None)


def scenario_build_balanced7_embeddings():
    print 'reading data'
    x, y = utils.read_data('resources/test/balanced7X.p', 'resources/test/balanced7y.p', 1)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    g.learn_embeddings(walks, 128, 'resources/test/balanced7.emd')
    print datetime.now() - s, ' s'
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv('resources/test/balanced7_walks.csv', index=False, header=None)


def scenario_vary_window_embeddings():
    print 'reading data'
    x, y = utils.read_data('resources/test/balanced7_100_thresh_X.p', 'resources/test/balanced7_100_thresh_y.p', 1)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    df = pd.DataFrame(walks)
    df.to_csv('resources/test/balanced7_100_thresh_walks.csv', index=False, header=None)
    for walk_length in np.arange(1, 10):
        print 'embedding window length {}'.format(walk_length)
        g.learn_embeddings(walks, 64, 'resources/test/balanced7_d64_window' + str(walk_length) + '.emd', walk_length)
        print datetime.now() - s, ' s'
        print walks.shape


def scenario_vary_window_2step_embeddings():
    print 'reading data'
    x, y = utils.read_data('resources/test/balanced7_100_thresh_X.p', 'resources/test/balanced7_100_thresh_y.p', 1)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 160)
    print 'generated walks of shape', walks.shape
    # only keep walks that land on the fans
    walks2 = walks[:, np.arange(walks.shape[1]) % 2 == 0]
    print 'generated 2 step walks of shape', walks2.shape
    df = pd.DataFrame(walks)
    df.to_csv('resources/test/balanced7_100_thresh_walks160.csv', index=False, header=None)
    df2 = pd.DataFrame(walks2)
    df2.to_csv('resources/test/balanced7_100_thresh_2step_walks.csv', index=False, header=None)
    for walk_length in np.arange(1, 10):
        print 'embedding window length {}'.format(walk_length)
        g.learn_embeddings(walks2, 128, 'resources/test/balanced7_2step_window' + str(walk_length) + '.emd',
                           walk_length)
        g.learn_embeddings(walks, 128, 'resources/test/balanced7_len160_window' + str(walk_length) + '.emd',
                           walk_length)
        print datetime.now() - s, ' s'


def generate_2step_embedding(window_size):
    s = datetime.now()
    walks_df = pd.read_csv('resources/test/balanced7_100_thresh_2step_walks.csv', header=None)
    walks = walks_df.values
    print 'embedding window length {}'.format(window_size)
    g = BipartiteGraph()
    g.learn_embeddings(walks, 128, 'resources/test/balanced7_2step_window' + str(window_size) + '.emd',
                       window_size)
    print datetime.now() - s, ' s'


def generate_balanced7_embedding(window_size):
    s = datetime.now()
    walks_df = pd.read_csv('resources/test/balanced7_100_thresh_walks.csv', header=None)
    walks = walks_df.values
    print 'embedding window length {}'.format(window_size)
    g = BipartiteGraph()
    g.learn_embeddings(walks, 128, 'resources/test/balanced7_window' + str(window_size) + '.emd',
                       window_size)
    print datetime.now() - s, ' s'


def generate_balanced7_10_thresh_embedding(window_size):
    s = datetime.now()
    walks_df = pd.read_csv('resources/test/balanced7_10_thresh_window_' + str(window_size) + 'walks.csv', header=None)
    walks = walks_df.values
    print 'embedding window length {}'.format(window_size)
    g = BipartiteGraph()
    g.learn_embeddings(walks, 128, 'resources/test/balanced7_10_thresh_window_' + str(window_size) + '.emd',
                       window_size)
    print datetime.now() - s, ' s'


def scenario_build_balanced7_10_thresh_embeddings():
    print 'reading data'
    x, y = utils.read_data('resources/test/balanced7_10_thresh_X.p', 'resources/test/balanced7_10_thresh_y.p', 0)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    g.learn_embeddings(walks, 128, 'resources/test/balanced7_10_thresh.emd')
    print datetime.now() - s, ' s'
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv('resources/test/balanced7_10_thresh_walks.csv', index=False, header=None)


def scenario_build_income_embeddings():
    print 'reading data'
    x, y = utils.read_data('local_resources/Socio_economic_classification_data/income_dataset/X_thresh10.p',
                           'local_resources/Socio_economic_classification_data/income_dataset/y_thresh10.p', 0)
    s = datetime.now()
    g = BipartiteGraph(x)
    print 'building edges'
    g.build_edge_array()
    print 'generating walks'
    walks = g.generate_walks(10, 80)
    g.learn_embeddings(walks, 64,
                       'local_resources/Socio_economic_classification_data/income_dataset/thresh10_64.emd')
    print datetime.now() - s, ' s'
    print walks.shape
    df = pd.DataFrame(walks)
    df.to_csv('local_resources/Socio_economic_classification_data/income_dataset/thresh10_walks.csv', index=False,
              header=None)


def scenario_build_different_size_income_embeddings():
    print 'reading data'
    x, y = utils.read_data('local_resources/Socio_economic_classification_data/income_dataset/X_thresh10.p',
                           'local_resources/Socio_economic_classification_data/income_dataset/y_thresh10.p', 0)
    s = datetime.now()
    g = BipartiteGraph(x)
    # print 'building edges'
    # g.build_edge_array()
    walks = pd.read_csv('local_resources/Socio_economic_classification_data/income_dataset/thresh10_walks.csv',
                        header=None).values
    sizes = [16, 32, 64, 128]
    for size in sizes:
        g.learn_embeddings(walks, size,
                           'local_resources/Socio_economic_classification_data/income_dataset/thresh10_' + str(
                               size) + '.emd')
        print 'size ', str(size), ' embeddings generated in ', datetime.now() - s, ' s'


def reindex_embeddings():
    """
    changes the first column of embeddings from an index to a Twitter ID
    :return:
    """
    y_path = 'local_resources/Socio_economic_classification_data/income_dataset/y_thresh10.p'
    target = utils.read_target(y_path)
    sizes = [16, 32, 64, 128]

    for size in sizes:
        print 'running embeddings of size ', size
        emd_path = 'local_resources/Socio_economic_classification_data/income_dataset/thresh10_{0}.emd'.format(size)
        x = utils.read_embedding(emd_path, target)
        df = pd.DataFrame(data=x, index=target.index)
        try:
            del df.index.name
        except AttributeError:
            pass
        df.to_csv(emd_path)


if __name__ == '__main__':
    scenario_build_different_size_income_embeddings()
    reindex_embeddings()
