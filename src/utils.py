import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from config import basedir
sns.set()


def convert_to_num(pd_series):
    pd_series = pd.to_numeric(pd_series, errors='coerce')
    pd_series = pd_series.replace([np.inf, -np.inf], np.nan)
    return pd_series


def woe(df, var_name, good_bad_var, discrete=True):
    df = pd.concat([df[var_name], df[good_bad_var]], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = (df['prop_good'] * df['n_obs'])
    df['n_bad'] = ((1 - df['prop_good']) * df['n_obs'])
    df['prob_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prob_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['woe'] = np.log(df['prob_n_good'] / df['prob_n_bad'])
    if discrete:
        df = df.sort_values(['woe'])
        df = df.reset_index(drop=True)
    df['diff_prob_good'] = df['prob_n_good'].diff().abs()
    df['diff_woe'] = df['woe'].diff().abs()
    df['IV'] = ((df['prob_n_good'] - df['prob_n_bad']) * df['woe']).sum()
    return df


def plot_woe(df_woe, rotation=0):
    X = np.array(df_woe.iloc[:, 0].apply(str))
    y = df_woe["woe"]
    plt.figure(figsize=(18, 6))
    plt.plot(X, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_woe.columns[0])
    plt.ylabel("Weight of Evidence")
    plt.title("Weight of Evidence by " + df_woe.columns[0].upper())
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, 'results', 'weight_of_evidence', df_woe.columns[0] + '.png'))


def print_test_results(print_str, df):
    print_str = print_str + " " * (50 - len(print_str))
    print(print_str, f"default rate = {calc_default(df)}%", "\t", f"return = {calc_annual_return(df)}%", "\t",
          f"no of loans = {len(df)}")


def calc_default(df):
    return np.round(1 - np.sum(df["good_bad"] / len(df)) * 100, 1)


def calc_annual_return(df):
    funded_amnt = df['funded_amnt'].sum()
    total_pymnt = df['total_pymnt'].sum()
    avg_term = (df['term'].mean() + 1) / 12.0
    return np.round(100 / np.power(funded_amnt / total_pymnt, 1 / (avg_term / 2)) - 100, 2)


def calc_annual_return_vec(df):
    funded_amnt = df['funded_amnt']
    total_pymnt = df['total_pymnt']
    avg_term = (df['term'] + 1) / 12.0
    return np.round(100 / np.power(funded_amnt / total_pymnt, 1 / (avg_term / 2)) - 100, 2)


def plot_histogram(df, column, rotation=0):

    x = df[column].sort_values()

    plt.figure()
    if len(x.unique()) > 50:
        sns.histplot(x, bins=20)
    else:
        sns.countplot(x)
    plt.title('Histogram of ' + column)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, 'results', 'histogram', column + '.png'))


def def_rates_by_categorical(df, column, sort=True):
    grouped = df.groupby([column, 'good_bad'])
    def_counts = grouped['loan_amnt'].count().unstack()
    N = def_counts.sum(axis=1)
    props = def_counts[0] / N
    if sort:
        props = props.sort_values()
    plt.figure()
    ax = props.plot(kind='bar')
    ax.set_ylabel("Default rate")
    ax.set_title("Default rates by {}".format(column))
    ax.set_xlabel(column)
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, 'results', 'plots', 'defaultRate_' + column + '.png'))
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
    plt.savefig(os.path.join(basedir, 'results', 'plots', 'interestRate_' + column + '.png'))
    plt.close()


def returns_by_categorical(df, column, with_variance=False, sort=True):
    returns = calc_annual_return_vec(df)
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
    plt.savefig(os.path.join(basedir, 'results', 'plots', 'returns_' + column + '.png'))
    plt.close()


def def_rates_by_hist(df, column, bin_idx):
    df = df.sort_values(column)
    grouped = df.groupby([column, 'good_bad'])
    def_counts = grouped[column].count().unstack()
    def_counts = def_counts.fillna(value=0)

    x = np.array(def_counts.index)
    defaults = np.array(def_counts[0])
    total = np.array(def_counts[1]) + defaults

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
    plt.savefig(os.path.join(basedir, 'results', 'plots', 'histogram_' + column + '.png'))

    plt.figure()
    plt.plot(x_new, no_rate)
    plt.title("Default rates by {}".format(column))
    plt.xlabel(column)
    plt.ylabel("Default rate")
    plt.tight_layout()
    plt.savefig(os.path.join(basedir, 'results', 'plots', 'defaultRate_' + column + '.png'))


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
    elif x == 'None':
        return -1
    else:
        return int(str(x).split(" ")[0].split("+")[0])


def process_home_ownership(x):
    if x == 'ANY':
        return 'RENT'
    elif x == 'OTHER':
        return 'RENT'
    elif x == 'NONE':
        return 'RENT'
    else:
        return x
