import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    # Read the datasets
    X_train = pd.read_csv('../data/processed/X_train.csv', sep=";")
    y_train = pd.read_csv('../data/processed/y_train.csv', sep=";")
    X_test = pd.read_csv('../data/processed/X_test.csv', sep=";")
    y_test = pd.read_csv('../data/processed/y_test.csv', sep=";")

    reg = LogisticRegression()

    reg.fit(X_train, y_train)


