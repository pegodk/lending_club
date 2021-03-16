import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_curve, roc_auc_score
from src.utils import calc_annual_return_vec, print_test_results
from config import basedir


if __name__ == "__main__":

    # Read the datasets
    X_train = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'X_train_continuous.csv'), sep=";")
    y_train = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'y_train.csv'), sep=";")
    X_test = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'X_test_continuous.csv'), sep=";")
    y_test = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'y_test.csv'), sep=";")

    print('Length of training set:', len(y_train))
    print('Length of testing set: ', len(y_test))

    ####################################################################################################################
    ######################################      Random Forest Classification      ######################################
    ####################################################################################################################

    reg = RandomForestClassifier()
    # reg = GradientBoostingRegressor()
    # reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

    reg.fit(X_train, y_train)
    y_train_predict = np.round(reg.predict(X_train), 2)
    y_test_predict = np.round(reg.predict(X_test), 2)

    y_hat_test = reg.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_hat_test))

    y_hat_test_proba = reg.predict_proba(X_test)[:][:, 1]
    predictions = pd.concat([y_test.reset_index(drop=True), pd.DataFrame(y_hat_test_proba)], axis=1)
    predictions.columns = ["y_test", "y_hat_test_proba"]

    fpr, tpr, thresholds = roc_curve(y_test, y_hat_test_proba)
    auc = roc_auc_score(y_test, y_hat_test_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr, linestyle="--", color="k")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve (AUC = {np.round(auc, 2)})")
    plt.savefig('../results/PD_RandomForest_model_auc.png')
    plt.show()

    scores = mean_absolute_error(y_test_predict, y_test)
    print('Mean Abs Error: {:.2f}'.format(scores))

    ####################################################################################################################
    ###########################################      Feature Importance      ###########################################
    ####################################################################################################################

    print_FeatureImportance = True
    if print_FeatureImportance:
        importances = reg.feature_importances_
        std = np.std([tree.feature_importances_ for tree in reg.estimators_], axis=0)
        indices = np.flip(np.argsort(importances), axis=0)
        xaxis = np.linspace(0, len(indices) - 1, len(indices))
        names = []
        for idx in indices:
            names.append(X_train.columns[idx])

        ax = plt.figure()
        plt.title("Feature Importance")
        plt.bar(xaxis, importances[indices] * 100, color="r", yerr=std[indices] * 100, align="center")
        plt.xticks(xaxis, names, rotation=90)
        plt.ylabel('%')
        plt.tight_layout()
        plt.savefig('../results/plots/FeatureImportance.png')

    ####################################################################################################################
    #######################################      Evaluating Output Results      ########################################
    ####################################################################################################################

    print_results = False
    if print_results:
        idx = y_test_predict > 15.0
        print_test_results(f"Yield (15%  < predict):", X_test[idx])

        idx = np.logical_and(y_test_predict > 10.0, y_test_predict < 15.0)
        print_test_results(f"Yield (10%  < predict < 15%):", X_test[idx])

        idx = np.logical_and(y_test_predict > 5.0, y_test_predict < 10.0)
        print_test_results(f"Yield (5%   < predict < 10%):", X_test[idx])

        idx = np.logical_and(y_test_predict > 0.0, y_test_predict < 5.0)
        print_test_results(f"Yield (0%   < predict < 5%):", X_test[idx])

        idx = np.logical_and(y_test_predict > -10.0, y_test_predict < 0.0)
        print_test_results(f"Yield (-10% < predict < 0%):", X_test[idx])

        idx = np.logical_and(y_test_predict > -20.0, y_test_predict < -10.0)
        print_test_results(f"Yield (-20% < predict < -10%):", X_test[idx])

        idx = y_test_predict < -20.0
        print_test_results(f"Yield (-20% > predict):", X_test[idx])

    plt.show(block=True)
