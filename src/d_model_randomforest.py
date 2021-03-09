import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.utils import calc_CAGR_vec, process_emp_length, process_home, print_test_results

if __name__ == "__main__":

    domain_columns = ["loan_amnt",
                      "funded_amnt",
                      "term",
                      "int_rate",
                      "installment",
                      "sub_grade",
                      # "emp_title",
                      "emp_length",
                      "home_ownership",
                      "annual_inc",
                      # "issue_d",
                      # "purpose",
                      # "title",
                      # "addr_state",
                      "dti",
                      # "earliest_cr_line",
                      "fico_range_low",
                      "fico_range_high",
                      "total_acc",
                      "total_pymnt",
                      "default"]

    # Read the datasets
    df = pd.read_csv('../data/processed/dataset_cleaned.csv', usecols=domain_columns)

    # Process features into numerical values
    df['emp_length'] = df['emp_length'].apply(process_emp_length)
    subgrade_sorted = sorted(np.unique(df["sub_grade"]))
    df['sub_grade'] = [subgrade_sorted.index(subgrade) for subgrade in df['sub_grade']]
    df['home_ownership'] = df['home_ownership'].apply(process_home)

    # TODO: Investigate problem with NaN
    df.dropna(inplace=True)

    # Split dataset into train and test
    X = df[domain_columns]
    y = calc_CAGR_vec(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print('Length of training set:', len(y_train))
    print('Length of testing set: ', len(y_test))

    ####################################################################################################################
    ######################################      Random Forest Classification      ######################################
    ####################################################################################################################

    reg = RandomForestRegressor()
    # reg = GradientBoostingRegressor()
    # reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

    reg.fit(X_train.drop(columns=['total_pymnt', 'default']), y_train)
    y_train_predict = np.round(reg.predict(X_train.drop(columns=['total_pymnt', 'default'])), 2)
    y_test_predict = np.round(reg.predict(X_test.drop(columns=['total_pymnt', 'default'])), 2)

    # reg.fit(X_train, y_train)
    # y_train_predict  = np.round(reg.predict(X_train), 2)
    # y_test_predict   = np.round(reg.predict(X_test), 2)

    # print(pd.DataFrame(data={'int_rate': X['int_rate'], 'outcome': y}))
    # print(pd.DataFrame(data={'int_rate': X_train['int_rate'], 'outcome': y_train, 'predict': y_train_predict}))
    # print(pd.DataFrame(data={'int_rate': X_test['int_rate'], 'outcome': y_test, 'predict': y_test_predict}))

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
            names.append(domain_columns[idx])

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

    print_results = True
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
