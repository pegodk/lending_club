import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_curve, roc_auc_score
from src.utils import calc_annual_return_vec, print_test_results
from config import basedir


if __name__ == "__main__":

    # Read the datasets
    train = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'PD_train_continuous.csv'), sep=";")
    test = pd.read_csv(os.path.join(basedir, 'data', 'processed', 'PD_test_continuous.csv'), sep=";")

    print('Length of training set:', len(train))
    print('Length of testing set: ', len(test))

    ####################################################################################################################
    ######################################      Random Forest Classification      ######################################
    ####################################################################################################################

    reg = GradientBoostingClassifier()

    reg.fit(train.drop(columns="good_bad"), train[["good_bad"]])
    y_train_predict = np.round(reg.predict(train.drop(columns="good_bad")), 2)
    y_test_predict = np.round(reg.predict(test.drop(columns="good_bad")), 2)

    y_hat_test = reg.predict(test.drop(columns="good_bad"))
    print("Accuracy:", accuracy_score(test[["good_bad"]], y_hat_test))

    y_hat_test_proba = reg.predict_proba(test)[:][:, 1]
    predictions = pd.concat([test[["good_bad"]].reset_index(drop=True), pd.DataFrame(y_hat_test_proba)], axis=1)
    predictions.columns = ["y_test", "y_hat_test_proba"]

    fpr, tpr, thresholds = roc_curve(test[["good_bad"]], y_hat_test_proba)
    auc = roc_auc_score(test[["good_bad"]], y_hat_test_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot(fpr, fpr, linestyle="--", color="k")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve (AUC = {np.round(auc, 2)})")
    plt.savefig('../results/PD_GradientBoosting_model_auc.png')
    plt.savefig(os.path.join(basedir, 'results', 'roc', 'PD_GradientBoosting.png'))
    plt.show()

    scores = mean_absolute_error(y_test_predict, test[["good_bad"]])
    print('Mean Abs Error: {:.2f}'.format(scores))

    ####################################################################################################################
    ###########################################      Feature Importance      ###########################################
    ####################################################################################################################

    print_FeatureImportance = False
    if print_FeatureImportance:
        importances = reg.feature_importances_
        std = np.std([tree.feature_importances_ for tree in reg.estimators_], axis=0)
        indices = np.flip(np.argsort(importances), axis=0)
        xaxis = np.linspace(0, len(indices) - 1, len(indices))
        names = []
        for idx in indices:
            names.append(train.columns[idx])

        ax = plt.figure()
        plt.title("Feature Importance")
        plt.bar(xaxis, importances[indices] * 100, color="r", yerr=std[indices] * 100, align="center")
        plt.xticks(xaxis, names, rotation=90)
        plt.ylabel('%')
        plt.tight_layout()
        plt.savefig(os.path.join(basedir, 'results', 'roc', 'PD_GradientBoosting_FeatureImportance.png'))

    ####################################################################################################################
    #######################################      Evaluating Output Results      ########################################
    ####################################################################################################################

    print_results = False
    if print_results:
        idx = y_test_predict > 15.0
        print_test_results(f"Yield (15%  < predict):", test[idx])

        idx = np.logical_and(y_test_predict > 10.0, y_test_predict < 15.0)
        print_test_results(f"Yield (10%  < predict < 15%):", test[idx])

        idx = np.logical_and(y_test_predict > 5.0, y_test_predict < 10.0)
        print_test_results(f"Yield (5%   < predict < 10%):", test[idx])

        idx = np.logical_and(y_test_predict > 0.0, y_test_predict < 5.0)
        print_test_results(f"Yield (0%   < predict < 5%):", test[idx])

        idx = np.logical_and(y_test_predict > -10.0, y_test_predict < 0.0)
        print_test_results(f"Yield (-10% < predict < 0%):", test[idx])

        idx = np.logical_and(y_test_predict > -20.0, y_test_predict < -10.0)
        print_test_results(f"Yield (-20% < predict < -10%):", test[idx])

        idx = y_test_predict < -20.0
        print_test_results(f"Yield (-20% > predict):", test[idx])

    plt.show(block=True)
