import os
import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from config import basedir
sns.set()


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
    train = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'PD_train_onehot.csv'), sep=";")
    test = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'PD_test_onehot.csv'), sep=";")

    target_var = "good_bad"

    # Define model and fit to training data
    reg = LogisticRegression(max_iter=10000)
    reg.fit(train.drop(columns=target_var), train[target_var])

    summary_table = pd.DataFrame(columns=["feature"], data=train.columns.values)
    summary_table["coefficient"] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ["intercept", reg.intercept_[0]]
    summary_table = summary_table.sort_index()
    summary_table.to_csv('../results/SummaryTable_LogisticRegression.csv', sep=";", index=False)
    # print(summary_table)

    y_hat_test = reg.predict(test.drop(columns=target_var))
    print("Accuracy:", accuracy_score(test[target_var], y_hat_test))

    y_hat_test_proba = reg.predict_proba(test.drop(columns=target_var))[:][:, 1]
    predictions = pd.concat([test[target_var].reset_index(drop=True), pd.DataFrame(y_hat_test_proba)], axis=1)
    predictions.columns = ["y_test", "y_hat_test_proba"]

    fpr, tpr, thresholds = roc_curve(test[target_var], y_hat_test_proba)
    auc = roc_auc_score(test[target_var], y_hat_test_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr, linestyle="--", color="k")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve (AUC = {np.round(auc, 2)})")
    plt.savefig(os.path.join(basedir, 'results', 'roc', 'PD_LogisticRegression.png'))
    plt.show()
