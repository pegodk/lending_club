import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import basedir


def label_encoder(X_train, X_test, categorical_cols):
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    for col in categorical_cols:
        encoder = LabelEncoder()
        X_train_enc[col] = encoder.fit_transform(X_train[col])
        X_test_enc[col] = encoder.transform(X_test[col])
    return X_train_enc, X_test_enc


if __name__ == "__main__":
    # Read the datasets
    df_train = pd.read_csv(os.path.join(basedir, 'data', 'temp', 'dataset_train.csv'))
    df_test = pd.read_csv(os.path.join(basedir, 'data', 'temp', 'dataset_test.csv'))

    input_vars = ["loan_amnt",
                  "term",
                  "int_rate",
                  "income_to_installment",
                  "grade",
                  "emp_length",
                  "home_ownership",
                  "annual_inc",
                  "purpose",
                  "addr_state",
                  "dti",
                  "fico_range_low"]

    target_var = ["good_bad"]

    X_train = df_train[input_vars]
    y_train = df_train[target_var]
    X_test = df_test[input_vars]
    y_test = df_test[target_var]

    X_train, X_test = label_encoder(X_train, X_test, categorical_cols=["grade",
                                                                       "home_ownership",
                                                                       "purpose",
                                                                       "addr_state"])

    # Show NaNs in dataset
    print(X_train[X_train.isna().any(axis=1)].to_string())

    # Fill NaNs
    X_train.replace([-np.inf, np.inf], np.nan, inplace=True)
    X_test.replace([-np.inf, np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    print(X_train[:10].to_string())
    print(X_test[:10].to_string())

    X_train.to_csv(os.path.join(basedir, 'data', 'processed', 'X_train_continuous.csv'), index=False, sep=";")
    y_train.to_csv(os.path.join(basedir, 'data', 'processed', 'y_train.csv'), index=False, sep=";")
    X_test.to_csv(os.path.join(basedir, 'data', 'processed', 'X_test_continuous.csv'), index=False, sep=";")
    y_test.to_csv(os.path.join(basedir, 'data', 'processed', 'y_test.csv'), index=False, sep=";")
