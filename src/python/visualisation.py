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

    x_path = '../../local_resources/income_dataset/X_thresh10.p'
    y_path = '../../local_resources/income_dataset/y_thresh10.p'
    emd_path = '../../local_results/dimension_32_num_10_length_80_context_10.emd'
    outpath = '../../local_results/figures/tsne.pdf'

    X, y = utils.read_data(x_path, y_path, threshold=10)

    target = utils.read_target(y_path)
    emd = pd.read_csv(emd_path, header=None, index_col=0, skiprows=1, sep=",")
    embedding = model.fit_transform(emd)

    # sb.set_context("notebook", font_scale=1.1)
    sns.set_style("ticks")

    print 'embedding shape is ', embedding.shape

    df = pd.DataFrame(data=embedding, index=None, columns=['x', 'y'])
    labels = np.array(target.loc[emd.index].mean_income)
    df['label'] = labels

    plot = sns.lmplot('x', 'y',
                      data=df,
                      fit_reg=False,
                      hue="label",
                      scatter_kws={"marker": "D",
                                   "s": 100})

    plot.savefig(outpath)


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
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colours, alpha=0.4)
    vert_labs = xrange(1, len(labels) + 1)
    for vert_lab, x, y in zip(vert_labs, embedding[:, 0], embedding[:, 1]):
        plt.annotate(
            vert_lab,
            xy=(x, y), xytext=(-2, 2),
            textcoords='offset points', ha='right', va='bottom', fontsize=8)
    plt.savefig(path)
    plt.clf()


if __name__ == '__main__':
    tsne_plot()
    # paths = ['results/age/graphs/avg_macro0.920170108-213545.csv', 'results/age/graphs/avg_micro0.920170108-213545.csv']
    # f1_line_plots(paths)
