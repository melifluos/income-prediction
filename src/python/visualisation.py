"""
Visualise age data
"""

import utils
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def tsne_plot():
    model = TSNE(n_components=2, random_state=0)

    x_path = 'resources/test/X.p'
    y_path = 'resources/test/y.p'
    emd_path = 'resources/test/test64.emd'

    X, y = utils.read_data(x_path, y_path, threshold=10)

    target = utils.read_target(y_path)
    X1 = utils.read_embedding(emd_path, target, 64)
    embedding = model.fit_transform(X1)

    # sb.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    print X1.shape

    df = pd.DataFrame(data=embedding, index=None, columns=['x', 'y'])
    df['label'] = y

    sns.lmplot('x', 'y',
               data=df,
               fit_reg=False,
               hue="label",
               scatter_kws={"marker": "D",
                            "s": 100})


def f1_line_plots(paths):
    """
    generate macro and micro F1 line plots
    :param paths: paths to the results csv tables
    :return: None
    """
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True)
    macro_df = pd.read_csv(paths[0], index_col=0)
    micro_df = pd.read_csv(paths[1], index_col=0)
    macro_df.transpose().plot(ax=axarr[0])
    axarr[0].set_ylabel('macro F1')
    micro_df.transpose().plot(ax=axarr[1])
    axarr[1].set_ylabel('micro F1')
    axarr[1].set_xlabel('% of labelled data')
    f.savefig('results/age/graphs/f1_line_plots' + utils.get_timestamp() + '.pdf')


def plot_embedding(embedding, labels, path):
    colours = labels
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colours, alpha=0.5)
    vert_labs = xrange(1, len(labels) + 1)
    for vert_lab, x, y in zip(vert_labs, embedding[:, 0], embedding[:, 1]):
        plt.annotate(
            vert_lab,
            xy=(x, y), xytext=(-2, 2),
            textcoords='offset points', ha='right', va='bottom', fontsize=8) \
            # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.savefig(path)
    plt.clf()


if __name__ == '__main__':
    paths = ['results/age/graphs/avg_macro0.920170108-213545.csv', 'results/age/graphs/avg_micro0.920170108-213545.csv']
    f1_line_plots(paths)
