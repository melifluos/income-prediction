"""
Runs a set of candidate detectors on the age data
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest, f_classif, RFECV
from sklearn import svm
import utils
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from age_detector import AgeDetector

__author__ = 'benchamberlain'

names = [
    "Logistic_Regression",
    # "Nearest_Neighbors",
    # "Linear_SVM",
    # "RBF_SVM",
    # "Decision_Tree",
    # "Random_Forest"
    # "AdaBoost",
    # "Gradient_Boosted_Tree"
]

names64 = [
    "Logistic_Regression64",
    # "Nearest_Neighbors64",
    # "Linear_SVM64",
    # "RBF_SVM64",
    # "Decision_Tree64",
    # "Random_Forest64"
    # "AdaBoost64",
    # "Gradient_Boosted_Tree64"
]

names128 = [
    "Logistic_Regression128",
    # "Nearest_Neighbors128",
    # "Linear_SVM128",
    # "RBF_SVM128",
    # "Decision_Tree128",
    # "Random_Forest128"
    # "AdaBoost128",
    # "Gradient_Boosted_Tree128"
]

classifiers = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=1.8),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.0073),
    # SVC(kernel='rbf', gamma=0.011, C=9.0, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=18, n_estimators=50, criterion='gini', max_features=0.46, n_jobs=-1)
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_64 = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=3.4),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.11),
    # SVC(kernel='rbf', gamma=0.018, C=31, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=6, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.21,n_jobs=-1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]

classifiers_embedded_128 = [
    LogisticRegression(multi_class='multinomial', solver='lbfgs', n_jobs=1, max_iter=1000, C=3.9),
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.11),
    # SVC(kernel='rbf', gamma=0.029, C=27.4, class_weight='balanced'),
    # DecisionTreeClassifier(max_depth=5),
    # this uses a random forest where: each tree is depth 5, 20 trees, split on entropy, each split uses 10% of features,
    # all of the cores are used
    # RandomForestClassifier(max_depth=7, n_estimators=50, criterion='entropy', bootstrap=False, max_features=0.12,n_jobs = -1),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(n_estimators=100)
]


def run_detectors(X, y, names, classifiers, n_folds):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """
    temp = pd.DataFrame(np.zeros(shape=(len(names), n_folds)))
    temp.index = names
    results = (temp, temp.copy())
    for name, detector in zip(names, classifiers):
        y_pred, results = run_cv_pred(X, y, detector, n_folds, name, results)
        print name
        utils.get_metrics(y, y_pred)
    return results


def run_experiments(X, y, names, classifiers, n_reps, train_pct):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """
    temp = pd.DataFrame(np.zeros(shape=(len(names), n_reps)))
    temp.index = names
    results = (temp, temp.copy())
    for name, detector in zip(names, classifiers):
        print 'running ' + str(name) + ' dataset'
        results = evaluate_test_sample(X, y, detector, n_reps, name, results, train_pct)
    return results


def run_cv_pred(X, y, clf, n_folds, name, results):
    """
    Run n-fold cross validation returning a prediction for every row of X
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :param clf: The
    :param n_folds:
    :return:
    """
    # Construct a kfolds object
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    splits = skf.split(X, y)
    y_pred = y.copy()

    # Iterate through folds
    for idx, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf.fit(X_train, y_train)
        try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
            preds = clf.predict(X_test)
        except TypeError:
            preds = clf.predict(X_test.todense())
        macro, micro = utils.get_metrics(y[test_index], preds)
        results[0].loc[name, idx] = macro
        results[1].loc[name, idx] = micro
        y_pred[test_index] = preds

    return y_pred, results


def evaluate_test_sample(X, y, clf, nreps, name, results, train_pct):
    """
    Calculate results for this clf at various train / test split percentages
    :param X: features
    :param y: targets
    :param clf: detector
    :param nreps: number of random repetitions
    :param name: name of the detector
    :param results: A tuple of Pandas DataFrames containing (macro, micro) F1 results
    :param train_pct: The percentage of the data used for training
    :return: A tuple of Pandas DataFrames containing (macro, micro) F1 results
    """
    seed = 0
    for rep in range(nreps):
        # setting a random seed will cause the same sample to be generated each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_pct, random_state=seed, stratify=y)
        seed += 1
        clf.fit(X_train, y_train)
        try:  # Gradient boosted trees do not accept sparse matrices in the predict function currently
            preds = clf.predict(X_test)
        except TypeError:
            preds = clf.predict(X_test.todense())
        macro, micro = utils.get_metrics(y_test, preds)
        results[0].loc[name, rep] = macro
        results[1].loc[name, rep] = micro
    return results


def run_all_datasets(datasets, y, names, classifiers, n_folds):
    """
    Loop through a list of datasets running potentially numerous classifiers on each
    :param datasets: iterable of numpy (sparse) arrays
    :param y: numpy (sparse) array of shape = (n_data, n_classes) of (n_data, 1)
    :param names: iterable of classifier names
    :param classifiers:
    :param n_folds:
    :return: A tuple of pandas DataFrames for each dataset containing (macroF1, microF1)
    """
    results = []
    for data in zip(datasets, names):
        temp = run_detectors(data[0], y, data[1], classifiers, n_folds)
        results.append(temp)
    return results


def run_all_train_pct(datasets, y, names, classifiers, n_reps):
    """
    :param datasets:
    :param y:
    :param names:
    :param classifiers:
    :param n_folds:
    :return:
    """
    all_results = {}
    for train_size in xrange(1, 10, 1):
        train_pct = train_size / 10.0
        print 'training percentage =' + str(train_pct)
        results = []
        for data in zip(datasets, names):
            temp = run_experiments(data[0], y, data[1], classifiers, n_reps, train_pct)
            results.append(temp)
        all_results[train_pct] = results
    return all_results


def read_embeddings(paths, target_path, sizes):
    targets = utils.read_pickle(target_path)
    y = np.array(targets['cat'])
    all_data = []
    for elem in zip(paths, sizes):
        data = utils.read_embedding(elem[0], targets, size=elem[1])
        all_data.append(data)
    return all_data, y


def build_ensembles(data, groups):
    """
    generates ensembles by columnwise concatenating arrays from a list
    :param data: A list of numpy arrays
    :param groups: a list of lists of indices into group. Each sub list represents the data sets to group together
    eg. [[1,2], [1,2,3]] will create 2 ensembles, the first containing the first and second data sets etc.
    :return: A list of numpy arrays where each array is an input to a classifier
    """
    ensemble_output = []
    for group in groups:
        ensemble = None
        for count, idx in enumerate(group):
            if count == 0:
                ensemble = data[idx - 1]
            else:
                ensemble = np.concatenate((ensemble, data[idx - 1]), axis=1)
        ensemble_output.append(ensemble)

    return ensemble_output


def bipartite_scenario():
    paths = ['resources/test/test128.emd', 'resources/test/test1282.emd']

    names = [['logistic_theirs'],
             ['logistic_mine']]

    y_path = 'resources/test/y.p'

    sizes = [128, 128]
    X, y = read_embeddings(paths, y_path, sizes)
    n_folds = 5
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/age_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/age_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def ensemble_scenario():
    deepwalk_path = 'resources/test/test128.emd'

    names = [['logistic'], ['logistic_deepwalk'], ['ensemble']]
    y_path = 'resources/test/y.p'
    x_path = 'resources/test/X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=1)
    x_deepwalk = utils.read_embedding(deepwalk_path, target, 128)
    all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)

    n_folds = 5
    X = [x, x_deepwalk, all_features]
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/ensemble_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/ensemble_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced_ensemble_scenario(threshold):
    deepwalk_path = 'resources/test/balanced7.emd'

    names = [['logistic'], ['logistic_deepwalk'], ['ensemble']]
    y_path = 'resources/test/balanced7y.p'
    x_path = 'resources/test/balanced7X.p'

    target = utils.read_target(y_path)

    x, y = utils.read_data(x_path, y_path, threshold=threshold)
    x_deepwalk = utils.read_embedding(deepwalk_path, target, 128)
    all_features = np.concatenate((x.toarray(), x_deepwalk), axis=1)

    n_folds = 5
    X = [x, x_deepwalk, all_features]
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_ensemble_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_ensemble_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def two_step_scenario():
    det_names = [['1 step'], ['2 step']]
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = ['resources/test/balanced7_window10.emd', 'resources/test/balanced7_2step_window10.emd']
    sizes = [128] * 2
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    n_folds = 10
    results = run_all_datasets(x_emd, y, det_names, classifiers_embedded_128, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_window10_2step_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_window10_2step_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def LINE_scenario():
    names = [['logistic'], ['deepwalk'], ['LINE']]

    x_path = 'resources/test/balanced7_100_thresh_X.p'

    line_path = 'LINE/linux/vec_all.txt'
    y_path = 'resources/test/balanced7_100_thresh_y.p'

    targets = utils.read_pickle(y_path)

    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)

    deep_x = utils.read_embedding('resources/test/node2vec/1.0_1.0.emd', targets)

    line_x = utils.read_LINE_embedding(line_path, targets)
    results = run_all_datasets([x, deep_x, line_x], y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_100_thresh_LINE_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_100_thresh_LINE_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_macro_LINE' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_micro_LINE' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_pq_scenario():
    names = [['logistic']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = []
    for p in [0.25, 0.5, 1.0, 2.0, 4.0]:
        for q in [0.25, 0.5, 1.0, 2.0, 4.0]:
            embedding_paths.append('resources/test/node2vec/' + str(p) + '_' + str(q) + '.emd')
            names.append(['logistic_window' + str(p) + '_' + str(q)])

    sizes = [128] * len(embedding_paths)
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)
    X = [x] + x_emd
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results = utils.stats_test(all_results)
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_pq_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_pq_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


def balanced7_pq_best_scenario():
    names = [['logistic'], ['logistic1.0_1.0'], ['logistic2.0_0.25']]
    x_path = 'resources/test/balanced7_100_thresh_X.p'
    y_path = 'resources/test/balanced7_100_thresh_y.p'
    embedding_paths = ['resources/test/node2vec/1.0_1.0.emd', 'resources/test/node2vec/2.0_0.25.emd']

    sizes = [128] * len(embedding_paths)
    x_emd, y = read_embeddings(embedding_paths, y_path, sizes)
    n_folds = 10
    x, y = utils.read_data(x_path, y_path, threshold=1)
    X = [x] + x_emd
    results = run_all_datasets(X, y, names, classifiers, n_folds)
    all_results = utils.merge_results(results)
    results, tests = utils.stats_test(all_results)
    tests[0].to_csv('results/age/balanced7_100_thresh_pq_macro_pvalues' + utils.get_timestamp() + '.csv')
    tests[1].to_csv('results/age/balanced7_100_thresh_pq_micro_pvalues' + utils.get_timestamp() + '.csv')
    print 'macro', results[0]
    print 'micro', results[1]
    macro_path = 'results/age/balanced7_100_thresh_pq_best_macro' + utils.get_timestamp() + '.csv'
    micro_path = 'results/age/balanced7_100_thresh_pq_best_micro' + utils.get_timestamp() + '.csv'
    results[0].to_csv(macro_path, index=True)
    results[1].to_csv(micro_path, index=True)


if __name__ == "__main__":
    # karate_scenario()
    # generate_graphs_scenario()
    # balanced7_pq_best_scenario()
    balanced7_10_thresh_scenario()

