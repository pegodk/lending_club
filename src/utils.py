import re
import numpy as np
import pandas as pd
import datetime
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import itertools
re_not_decimal = re.compile(r'[^\.0-9]*')


def print_test_results(print_str, df):
    print_str = print_str + " " * (50 - len(print_str))
    print(print_str, f"default rate = {calc_default(df)}%", "\t", f"return = {calc_CAGR(df)}%", "\t",
          f"no of loans = {len(df)}")


def calc_default(df):
    return np.round(np.sum(df["default"] / len(df)) * 100, 1)


def calc_CAGR(df):
    funded_amnt = df['funded_amnt'].sum()
    total_pymnt = df['total_pymnt'].sum()
    avg_term = (df['term'].mean() + 1) / 12.0
    return np.round(100 / np.power(funded_amnt / total_pymnt, 1 / (avg_term / 2)) - 100, 2)


def calc_CAGR_vec(df):
    funded_amnt = df['funded_amnt']
    total_pymnt = df['total_pymnt']
    avg_term = (df['term'] + 1) / 12.0
    return np.round(100 / np.power(funded_amnt / total_pymnt, 1 / (avg_term / 2)) - 100, 2)


def def_rates_by_categorical(df, column, sort=True):
    grouped = df.groupby([column, 'loan_status'])
    def_counts = grouped['loan_amnt'].count().unstack()
    N = def_counts.sum(axis=1)
    props = def_counts['Charged Off'] / N
    if sort:
        props = props.sort_values()
    plt.figure()
    ax = props.plot(kind='bar')
    ax.set_ylabel("Default rate")
    ax.set_title("Default rates by {}".format(column))
    ax.set_xlabel(column)
    plt.tight_layout()
    plt.savefig('../results/plots/histogram_' + column + '.png')
    plt.close()


def int_rates_by_categorical(df, column, with_variance=False, sort=True):
    rates = df['int_rate']
    if sort:
        df = df.sort_values(by=column)
    labels = df[column].unique()
    means, vars = [[], []]
    for idx, value in enumerate(labels):
        idxs = df[column] == value
        means.append(rates[idxs].mean())
        vars.append(rates[idxs].std())
    dataframe = pd.Series(data=means, index=labels)

    plt.figure()
    if with_variance:
        ax = dataframe.plot(kind='bar', yerr=vars)
    else:
        ax = dataframe.plot(kind='bar')
    ax.set_ylabel("Interest rate (%)")
    ax.set_title("Interest rate by {}".format(column))
    ax.set_xlabel(column)
    plt.tight_layout()
    plt.savefig('../results/plots/interestRate_' + column + '.png')
    plt.close()


def calc_CAGR_vec(df):
    funded_amnt = df['funded_amnt']
    total_pymnt = df['total_pymnt']
    avg_term = (df['term'] + 1) / 12.0
    return np.round(100 / np.power(funded_amnt / total_pymnt, 1 / (avg_term / 2)) - 100, 2)


def returns_by_categorical(df, column, with_variance=False, sort=True):
    returns = calc_CAGR_vec(df)
    if sort:
        df = df.sort_values(by=column)
    labels = df[column].unique()
    means, vars = [[], []]
    for idx, value in enumerate(labels):
        idxs = df[column] == value
        means.append(returns[idxs].mean())
        vars.append(returns[idxs].var())

    dataframe = pd.Series(data=means, index=labels)

    plt.figure()
    if with_variance:
        ax = dataframe.plot(kind='bar', yerr=vars)
    else:
        ax = dataframe.plot(kind='bar')
    ax.set_ylabel("Returns (%)")
    ax.set_title("Returns by {}".format(column))
    ax.set_xlabel(column)
    plt.tight_layout()
    plt.savefig('../results/plots/returns_' + column + '.png')
    plt.close()


def def_rates_by_hist(df, column, bin_idx):
    df = df.sort_values(column)
    grouped = df.groupby([column, 'loan_status'])
    def_counts = grouped[column].count().unstack()
    def_counts = def_counts.fillna(value=0)

    x = np.array(def_counts.index)
    defaults = np.array(def_counts['Charged Off'])
    total = np.array(def_counts['Fully Paid']) + defaults

    no_default = np.zeros(np.size(bin_idx) - 1)
    no_total = np.zeros(np.size(bin_idx) - 1)

    for idx, value in enumerate(x):
        for idx2, value2 in enumerate(bin_idx[:-1]):
            if value > bin_idx[idx2] and value < bin_idx[idx2 + 1]:
                no_default[idx2] += defaults[idx]
                no_total[idx2] += total[idx]
                break

    x_new = []
    for idx, _ in enumerate(bin_idx[:-1]):
        x_new.append((bin_idx[idx] + bin_idx[idx + 1]) / 2)

    no_rate = no_default / no_total

    plt.figure()
    plt.hist(x, bins=bin_idx)
    plt.title('Histogram of ' + column)
    plt.tight_layout()
    plt.savefig('../results/plots/histogram_' + column + '.png')

    plt.figure()
    plt.plot(x_new, no_rate)
    plt.title("Default rates by {}".format(column))
    plt.xlabel(column)
    plt.ylabel("Default rate")
    plt.tight_layout()
    plt.savefig('../results/plots/defaultRate___' + column + '.png')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def process_emp_length(x):
    if x == "< 1 year":
        return 0
    elif x == "None":
        return -1
    else:
        return int(str(x).split(" ")[0].split("+")[0])


def process_home(x):
    if x == 'OWN':
        return 0
    if x == 'MORTGAGE':
        return 1
    if x == 'RENT':
        return 2
    if x == 'NONE':
        return 3
    return 4


def leaf_depths(tree, node_id=0):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child == _tree.TREE_LEAF:
        depths = np.array([0])
    else:
        left_depths = leaf_depths(tree, left_child) + 1
        right_depths = leaf_depths(tree, right_child) + 1
        depths = np.append(left_depths, right_depths)
    return depths


def leaf_samples(tree, node_id=0):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child == _tree.TREE_LEAF:
        samples = np.array([tree.n_node_samples[node_id]])
    else:
        left_samples = leaf_samples(tree, left_child)
        right_samples = leaf_samples(tree, right_child)
        samples = np.append(left_samples, right_samples)
    return samples


def draw_tree(ensemble, tree_id=0):
    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    tree = ensemble.estimators_[tree_id].tree_
    depths = leaf_depths(tree)
    plt.hist(depths, histtype='step', color='#9933ff',
             bins=range(min(depths), max(depths) + 1))
    plt.xlabel("Depth of leaf nodes (tree %s)" % tree_id)
    plt.subplot(212)
    samples = leaf_samples(tree)
    plt.hist(samples, histtype='step', color='#3399ff',
             bins=range(min(samples), max(samples) + 1))
    plt.xlabel("Number of samples in leaf nodes (tree %s)" % tree_id)
    plt.show()


def draw_ensemble(ensemble):
    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    depths_all = np.array([], dtype=int)
    for x in ensemble.estimators_:
        tree = x.tree_
        depths = leaf_depths(tree)
        depths_all = np.append(depths_all, depths)
        plt.hist(depths, histtype='step', color='#ddaaff', bins=range(min(depths), max(depths) + 1))
    plt.hist(depths_all, histtype='step', color='#9933ff', bins=range(min(depths_all), max(depths_all) + 1),
             weights=np.ones(len(depths_all)) / len(ensemble.estimators_), linewidth=2)
    plt.xlabel("Depth of leaf nodes")
    samples_all = np.array([], dtype=int)
    plt.subplot(212)
    for x in ensemble.estimators_:
        tree = x.tree_
        samples = leaf_samples(tree)
        samples_all = np.append(samples_all, samples)
        plt.hist(samples, histtype='step', color='#aaddff', bins=range(min(samples), max(samples) + 1))
    plt.hist(samples_all, histtype='step', color='#3399ff',
             bins=range(min(samples_all), max(samples_all) + 1),
             weights=np.ones(len(samples_all)) / len(ensemble.estimators_),
             linewidth=2)
    plt.xlabel("Number of samples in leaf nodes")
    plt.show()
