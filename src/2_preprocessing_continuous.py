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
                  "fico_range_low",
                  "good_bad"]

    PD_train = df_train[input_vars]
    PD_test = df_test[input_vars]

    PD_train, PD_test = label_encoder(PD_train, PD_test, categorical_cols=["grade",
                                                                       "home_ownership",
                                                                       "purpose",
                                                                       "addr_state"])

    # Show NaNs in dataset
    print(PD_train[PD_train.isna().any(axis=1)].to_string())

    # Fill NaNs
    PD_train.replace([-np.inf, np.inf], np.nan, inplace=True)
    PD_test.replace([-np.inf, np.inf], np.nan, inplace=True)
    PD_train.fillna(0, inplace=True)
    PD_test.fillna(0, inplace=True)

    print(PD_train[:10].to_string())
    print(PD_test[:10].to_string())

    PD_train.to_csv(os.path.join(basedir, 'data', 'processed', 'PD_train_continuous.csv'), index=False, sep=";")
    PD_test.to_csv(os.path.join(basedir, 'data', 'processed', 'PD_test_continuous.csv'), index=False, sep=";")
