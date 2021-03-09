import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    # Read the datasets
    df = pd.read_csv('../data/processed/dataset_cleaned.csv')

    # Calculate "issue year"
    df['issue_year'] = df['issue_d'].apply(lambda x: int(x.split('-')[1]))

    def_rates_by_categorical(df, 'purpose')
    def_rates_by_categorical(df, 'home_ownership')
    def_rates_by_categorical(df, 'grade')
    def_rates_by_categorical(df, 'emp_length')
    def_rates_by_categorical(df, 'issue_year')

    int_rates_by_categorical(df, 'grade', with_variance=True)
    int_rates_by_categorical(df, 'purpose', with_variance=True)

    returns_by_categorical(df, 'grade')
    returns_by_categorical(df, 'purpose')

    bin_idx = np.linspace(0, 250000, 40)
    def_rates_by_hist(df, 'annual_inc', bin_idx=bin_idx)

    bin_idx = np.linspace(5, 25, 40)
    def_rates_by_hist(df, 'int_rate', bin_idx=bin_idx)
