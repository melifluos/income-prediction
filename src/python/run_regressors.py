"""
A module to test the predictive performance of embeddings
"""

import pandas as pd
import utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.externals import joblib
from scipy import stats
import numpy as np
from math import sqrt
from scipy.stats import pearsonr

regressors = [
    # LinearRegression(),  can't handle categorical variables
    Ridge(alpha=300, solver='auto', max_iter=1000),
    # Lasso(alpha=10, solver='auto', max_iter=1000),
    RandomForestRegressor(max_depth=5, n_estimators=50, bootstrap=True, criterion='mse', max_features=0.1)
]


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
    kf = KFold(n_splits=n_folds, shuffle=True)
    splits = kf.split(X, y)
    y_pred = y.copy()
    # Iterate through folds
    for idx, (train_index, test_index) in enumerate(splits):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results.loc[name, idx] = mean_absolute_error(y[test_index], preds)
        y_pred[test_index] = preds
    # add on training F1
    clf.fit(X, y)
    preds = clf.predict(X)
    results.loc[name, n_folds] = mean_absolute_error(y, preds)
    y_pred[test_index] = preds

    return y_pred, results


def run_regressors(X, y, names, regressors, n_folds):
    """
    Runs a detector on the age data and returns accuracy
    :param X: A scipy sparse feature matrix
    :param y: The target labels corresponding to rows of X
    :return: The accuracy of the detector
    """
    results = pd.DataFrame(np.zeros(shape=(len(names), n_folds + 1)))  # leave space for the training error
    results.index = names
    for name, detector in zip(names, regressors):
        y_pred, results = run_cv_pred(X, y, detector, n_folds, name, results)
        print 'pearson correlation ', name, ' is: ', pearsonr(y, y_pred)
    return results


def run_all_datasets(datasets, y, names, regressors, n_folds):
    """
    Loop through a list of datasets running potentially numerous regressors on each
    :param datasets: iterable of numpy (sparse) arrays
    :param y: numpy (sparse) array of shape = (n_data, n_classes) of (n_data, 1)
    :param names: iterable of classifier names
    :param regressors: A list of intialised scikit-learn compatible regressors
    :param n_folds:
    :return: A tuple of pandas DataFrames for each dataset containing (macroF1, microF1)
    """
    results = []
    for data in zip(datasets, names):
        temp = run_regressors(data[0], y, data[1], regressors, n_folds)
        results.append(temp)
    return results


def reduce_features(features):
    """
    Use a pickled rfecv object to reduce the number of embedding features
    :return:
    """
    rfecv = joblib.load('../../local_resources/rfecv.pkl')
    features = rfecv.transform(features)
    return features


def reduce_embedding(embedding):
    data = reduce_features(embedding.values)
    return pd.DataFrame(index=embedding.index, data=data)


def t_tests(results):
    """
    performs a 2 sided t-test to see if difference in models is significant
    :param results_tuples: An array of pandas DataFrames (macro,micro)
    :return:
    """
    print results.head()
    results['mean'] = results.ix[:, :-2].mean(axis=1)
    results = results.sort('mean', ascending=True)

    try:
        print '1 versus 2'
        print(stats.ttest_ind(a=results.ix[0, 0:-2],
                              b=results.ix[1, 0:-2],
                              equal_var=False))
    except IndexError:
        pass

    try:
        print '2 versus 3'
        print(stats.ttest_ind(a=results.ix[1, 0:-2],
                              b=results.ix[2, 0:-2],
                              equal_var=False))
    except IndexError:
        pass

    try:
        print '3 versus 4'
        print(stats.ttest_ind(a=results.ix[1, 0:-2],
                              b=results.ix[2, 0:-2],
                              equal_var=False))
    except IndexError:
        pass

    tests = utils.t_grid(results)

    return results, tests


def neural_ltv_regression_cust_emd_scenario():
    scaler = StandardScaler()
    feature_path = '../../local_resources/features_1in10000.tsv'
    emd = pd.read_csv('../../local_results/prod2cust.emd', header=None, index_col=0, skiprows=1, sep=",")
    del emd.index.name

    all_feat = features.join(emd)
    X1 = features.values.astype(np.float)
    X1 = scaler.fit_transform(X1)
    X2 = all_feat.values.astype(np.float)
    X2 = scaler.fit_transform(X2)

    n_folds = 20
    results = run_all_datasets([X1, X2, emd.values], y, names, regressors, n_folds)
    # create a single data frame
    all_results = pd.concat([x for x in results])
    all_results.rename(columns={n_folds: 'train'}, inplace=True)
    results, tests = t_tests(all_results)
    print results
    path = '../../results/regression/train_100000' + utils.get_timestamp() + '.csv'
    results.to_csv(path, index=True)


def income_scenario():
    # names = np.array(
    #     [['ridge without emd', 'RF without emd'],
    #      ['ridge with emd', 'RF with emd'],
    #      ['ridge just emd', 'RF just emd']])
    # names = [['ridge', 'RF']]
    names = [['ridge']]
    y_path = '../local_resources/Socio_economic_classification_data/income_dataset/y_thresh10.p'
    emd_path = '../local_resources/Socio_economic_classification_data/income_dataset/thresh10_64.emd'

    target = utils.read_target(y_path)
    x = utils.read_embedding(emd_path, target)
    y = np.array(target['mean_income'])
    n_folds = 10
    # x, y = utils.read_data(x_path, y_path, threshold=1)
    results = run_all_datasets([x], y, names, regressors, n_folds)
    # all_results = utils.merge_results(results)
    all_results = pd.concat([x for x in results])
    all_results.rename(columns={n_folds: 'train'}, inplace=True)
    results, tests = t_tests(all_results)
    print results
    path = '../results/income/thresh10_' + utils.get_timestamp() + '.csv'
    results.to_csv(path, index=True)


def income_different_size_embedding_scenario():
    # names = np.array(
    #     [['ridge without emd', 'RF without emd'],
    #      ['ridge with emd', 'RF with emd'],
    #      ['ridge just emd', 'RF just emd']])
    # names = [['ridge', 'RF']]
    names = [['ridge']]
    y_path = '../local_resources/Socio_economic_classification_data/income_dataset/y_thresh10.p'

    target = utils.read_target(y_path)
    y = np.array(target['mean_income'])
    n_folds = 10
    sizes = [16, 32, 64, 128]
    for size in sizes:
        print 'running embeddings of size ', size
        emd_path = '../local_resources/Socio_economic_classification_data/income_dataset/thresh10_{0}.emd'.format(size)
        x = utils.read_embedding(emd_path, target)
        results = run_all_datasets([x], y, names, regressors, n_folds)
        # all_results = utils.merge_results(results)
        all_results = pd.concat([x for x in results])
        all_results.rename(columns={n_folds: 'train'}, inplace=True)
        results, tests = t_tests(all_results)
        print results
        path = '../results/income/thresh10_' + str(size) + '_' + utils.get_timestamp() + '.csv'
        results.to_csv(path, index=True)


def nikos_test_scenario():
    names = [['ridge']]
    y_path = '../local_resources/Socio_economic_classification_data/income_dataset/y_thresh10.p'
    target = utils.read_target(y_path)
    y = np.array(target['mean_income'])
    n_folds = 10
    sizes = [16, 32, 64, 128]
    for size in sizes:
        print 'running for size {} \n'.format(size)
        emd_path = '../local_resources/Socio_economic_classification_data/income_dataset/thresh10_{0}.emd'.format(size)
        x = pd.read_csv(emd_path, index_col=0)
        x = x.as_matrix()
        results = run_all_datasets([x], y, names, regressors, n_folds)
        all_results = pd.concat([x for x in results])
        all_results.rename(columns={n_folds: 'train'}, inplace=True)
        results, tests = t_tests(all_results)
        print results


if __name__ == '__main__':
    nikos_test_scenario()
