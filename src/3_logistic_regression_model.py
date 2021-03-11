import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy.stats as stat


class LogisticRegression_with_p_values:

    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


if __name__ == "__main__":
    # Read the datasets
    X_train = pd.read_csv('../data/processed/X_train.csv', sep=";", nrows=100000)
    y_train = pd.read_csv('../data/processed/y_train.csv', sep=";", nrows=100000)
    X_test = pd.read_csv('../data/processed/X_test.csv', sep=";", nrows=10000)
    y_test = pd.read_csv('../data/processed/y_test.csv', sep=";", nrows=10000)

    reg = LogisticRegression(max_iter=10000)

    reg.fit(X_train, y_train)

    print(reg.intercept_)
    print(reg.coef_)

    summary_table = pd.DataFrame(columns=["feature"], data=X_train.columns.values)
    summary_table["coefficient"] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ["intercept", reg.intercept_[0]]
    summary_table = summary_table.sort_index()
    summary_table.to_csv('../data/processed/summary_table.csv', sep=";", index=False)
    print(summary_table)
