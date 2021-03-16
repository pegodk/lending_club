import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config import basedir


def preprocess(train, test, continuous_vars):
    for col in continuous_vars:
        # Create bins based on training set
        train[col], bins = pd.qcut(train[col], q=10, retbins=True, labels=False)
        bins = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))

        # Apply bins from training set to test set
        test[col] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
    return train, test


def one_hot_encoder(X_train, X_test, target_var):
    X_train_enc = None
    X_test_enc = None

    for col in X_train.columns:

        if col == target_var:
            if X_train_enc is not None and X_test_enc is not None:
                X_train_enc = pd.concat([X_train_enc, X_train[[target_var]]], axis=1)
                X_test_enc = pd.concat([X_test_enc, X_test[[target_var]]], axis=1)
            else:
                X_train_enc = X_train[[target_var]]
                X_test_enc = X_test[[target_var]]

        else:

            # Fit encoder to training data
            encoder = OneHotEncoder()
            enc_train = encoder.fit_transform(X_train[[col]])
            enc_train = pd.DataFrame(enc_train.toarray(), columns=encoder.categories_)
            enc_train.columns = [col + ":" + str(new_col[0]) for new_col in enc_train.columns]

            # Apply fitted encoder to test data
            enc_test = encoder.transform(X_test[[col]])
            enc_test = pd.DataFrame(enc_test.toarray(), columns=encoder.categories_)
            enc_test.columns = [col + ":" + str(new_col[0]) for new_col in enc_test.columns]

            if X_train_enc is not None and X_test_enc is not None:
                X_train_enc = pd.concat([X_train_enc, enc_train], axis=1)
                X_test_enc = pd.concat([X_test_enc, enc_test], axis=1)
            else:
                X_train_enc = enc_train
                X_test_enc = enc_test
    return X_train_enc, X_test_enc


if __name__ == "__main__":
    # Read the datasets
    df_train = pd.read_csv(os.path.join(basedir, 'data', 'temp', 'dataset_train.csv'))
    df_test = pd.read_csv(os.path.join(basedir, 'data', 'temp', 'dataset_test.csv'))

    df_train, df_test = preprocess(df_train, df_test, continuous_vars=["loan_amnt", "int_rate", "annual_inc", "dti",
                                                                       "fico_range_low", "income_to_installment"])

    print(df_train[:5].to_string())
    print(df_test[:5].to_string())

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

    PD_train, PD_test = one_hot_encoder(PD_train, PD_test, target_var="good_bad")

    print(PD_train[:5].to_string())

    PD_train.to_csv(os.path.join(basedir, 'data', 'processed', 'PD_train_onehot.csv'), index=False, sep=";")
    PD_test.to_csv(os.path.join(basedir, 'data', 'processed', 'PD_test_onehot.csv'), index=False, sep=";")
