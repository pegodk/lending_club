import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from src.utils import calc_CAGR, calc_CAGR_vec, process_emp_length, process_home

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
                      "total_pymnt"]

    # Read the datasets
    df = pd.read_csv('../data/processed/dataset_cleaned.csv', usecols=domain_columns)

    # Process features into numerical values
    df['emp_length'] = df['emp_length'].apply(process_emp_length)
    subgrade_sorted = sorted(np.unique(df["sub_grade"]))
    df['sub_grade'] = [subgrade_sorted.index(subgrade) for subgrade in df['sub_grade']]
    df['home_ownership'] = df['home_ownership'].apply(process_home)

    # df['revol_util'] = df['revol_util'].apply(process_revol_util)
    # df['emp_title'] = df['emp_title'].fillna('None')
    # df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(0)
    # df['home_ownership2'] = df['home_ownership'].apply(process_home)
    #
    # # Construct new features
    # df['issue_year'] = df['issue_d'].apply(process_issueyear)
    # df['requested_minus_funded'] = df['loan_amnt'] - df['funded_amnt']
    # df['has_employer_info'] = df['emp_title'].isnull()
    # df['is_employed'] = df['emp_length'].isnull()
    # df['installment_over_income'] = df['installment'] * 12 / df['annual_inc']
    # df['debt_to_income'] = (df['revol_bal'] + df['funded_amnt']) / df['annual_inc']

    print(df[:10].to_string())
    df.fillna(0, inplace=True)

    # Split dataset into train and test
    X = df[domain_columns].copy()
    y = calc_CAGR_vec(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print('Length of training set:', len(y_train))
    print('Length of testing set: ', len(y_test))

    reg = RandomForestRegressor()
    # reg = GradientBoostingRegressor()
    # reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

    reg.fit(X_train.drop(columns=['total_pymnt']), y_train)
    y_train_predict = np.round(reg.predict(X_train.drop(columns=['total_pymnt'])), 2)
    y_test_predict = np.round(reg.predict(X_test.drop(columns=['total_pymnt'])), 2)

    # reg.fit(X_train, y_train)
    # y_train_predict  = np.round(reg.predict(X_train), 2)
    # y_test_predict   = np.round(reg.predict(X_test), 2)

    # print(pd.DataFrame(data={'int_rate': X['int_rate'], 'outcome': y}))
    # print(pd.DataFrame(data={'int_rate': X_train['int_rate'], 'outcome': y_train, 'predict': y_train_predict}))
    # print(pd.DataFrame(data={'int_rate': X_test['int_rate'], 'outcome': y_test, 'predict': y_test_predict}))

    scores = mean_absolute_error(y_test_predict, y_test)
    print('Mean Abs Error: {:.2f}'.format(scores))

    # ######################################################################################################################
    # ##############################################      Feature Tools      ###############################################
    # ######################################################################################################################

    # import featuretools as ft

    # # creating and entity set 'es'
    # es = ft.EntitySet(id = 'return')
    #
    # # adding a dataframe
    # es.entity_from_dataframe(entity_id = 'LendingClub', dataframe = X, index = 'id')
    #
    # print(es)

    # cutoff_times   = X['total_pymnt']
    #
    # agg_primitives = ['Sum', 'Std', 'Max', 'Min', 'Mean', 'Count', 'Percent_True', 'Num_Unique', 'Mode', 'Trend', 'Skew']
    #
    # feature_matrix, features = ft.dfs(
    #     entityset=es,
    #     target_entity="return",
    #     trans_primitives=[],
    #     agg_primitives=agg_primitives,
    #     max_depth=3,
    #     cutoff_time=cutoff_times,
    #     verbose=True)

    # print(X.head())
    # print(es)
    #
    # feature_matrix, feature_names = ft.dfs(entityset=es, target_entity = 'LendingClub', max_depth = 2, verbose = 1, n_jobs = 3)
    #
    # print(feature_names)
    # print(feature_matrix.columns)

    # ######################################################################################################################
    # ############################################      Feature Importance      ############################################
    # ######################################################################################################################

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

    # ######################################################################################################################
    # ########################################      Evaluating Output Results      #########################################
    # ######################################################################################################################

    # draw_tree(reg)

    # # Extract single tree
    # estimator = reg.estimators_[5]
    #
    # from sklearn.tree import export_graphviz
    # # Export as dot file
    # export_graphviz(estimator, out_file='tree.dot',
    #                 feature_names = domain_columns[:-1],
    #                 rounded = True, proportion = False,
    #                 precision = 2, filled = True)
    #
    # # Convert to png using system command (requires Graphviz)
    # from subprocess import call
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    # for tree_in_forest in reg.estimators_:
    #     export_graphviz(tree_in_forest, feature_names=X.drop(columns=['total_pymnt']).columns, filled=True, rounded=True, out_file='tree.dot')
    # os.system('dot -Tpng tree.dot -o tree.png')

    # tree_in_forest = reg.estimators_[5]
    # export_graphviz(tree_in_forest, feature_names=X.drop(columns=['total_pymnt']).columns, filled=True, rounded=True, out_file='tree.dot')
    # from subprocess import call
    # import pydot
    #
    # (graph,) = pydot.graph_from_dot_file('tree.dot')
    # graph.write_png('tree.png')

    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

    # ######################################################################################################################
    # ########################################      Evaluating Output Results      #########################################
    # ######################################################################################################################

    print_Results = True
    if print_Results:
        idx = y_test_predict > 15.0
        print('Yield (15%  < predict):\t\t\t\t', calc_CAGR(X_test[idx]), '%   \t\t', idx.sum(), '\tloans')

        idx = np.logical_and(y_test_predict > 10.0, y_test_predict < 15.0)
        print('Yield (10%  < predict < 15%):\t\t', calc_CAGR(X_test[idx]), '%   \t\t', idx.sum(), '\tloans')

        idx = np.logical_and(y_test_predict > 5.0, y_test_predict < 10.0)
        print('Yield (5%   < predict < 10%):\t\t', calc_CAGR(X_test[idx]), '%   \t\t', idx.sum(), '\tloans')

        idx = np.logical_and(y_test_predict > 0.0, y_test_predict < 5.0)
        print('Yield (0%   < predict < 5%):\t\t', calc_CAGR(X_test[idx]), '%   \t\t', idx.sum(), '\tloans')

        idx = np.logical_and(y_test_predict > -10.0, y_test_predict < 0.0)
        print('Yield (-10% < predict < 0%):\t\t', calc_CAGR(X_test[idx]), '%   \t\t', idx.sum(), '\tloans')

        idx = np.logical_and(y_test_predict > -20.0, y_test_predict < -10.0)
        print('Yield (-20% < predict < -10%):\t\t', calc_CAGR(X_test[idx]), '%   \t\t', idx.sum(), '\tloans')

        idx = y_test_predict < -20.0
        print('Yield (-20% > predict):\t\t\t\t', calc_CAGR(X_test[idx]), '%   \t\t', idx.sum(), '\tloans')

    plt.show(block=True)