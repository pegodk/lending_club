import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from src.utils import print_test_results, calc_annual_return_vec, weight_of_evidence, plot_woe
from src.utils import def_rates_by_categorical, int_rates_by_categorical, returns_by_categorical, def_rates_by_hist
from config import basedir


if __name__ == "__main__":
    # Read the datasets
    df = pd.read_csv(os.path.join(basedir, 'data', 'temp', 'dataset_train.csv'))

    # Calculate "issue year"
    df['issue_year'] = df['issue_d'].apply(lambda x: int(x.split('-')[1]))

    # Basic data analysis
    def_rates_by_categorical(df, 'purpose')
    def_rates_by_categorical(df, 'home_ownership')
    def_rates_by_categorical(df, 'grade')
    def_rates_by_categorical(df, 'emp_length')
    def_rates_by_categorical(df, 'issue_year')
    int_rates_by_categorical(df, 'grade', with_variance=True)
    int_rates_by_categorical(df, 'purpose', with_variance=True)
    returns_by_categorical(df, 'grade')
    returns_by_categorical(df, 'purpose')
    bin_idx = np.linspace(0, 250000, 40)
    def_rates_by_hist(df, 'annual_inc', bin_idx=bin_idx)
    bin_idx = np.linspace(5, 25, 40)
    def_rates_by_hist(df, 'int_rate', bin_idx=bin_idx)

    # Do fine classing of continuous variables
    df["loan_amnt"] = pd.qcut(df["loan_amnt"], 10)
    df["int_rate"] = pd.qcut(df["int_rate"], 20)
    df["annual_inc"] = pd.qcut(df["annual_inc"], 20)
    df["dti"] = pd.qcut(df["dti"], 20)
    df["fico_range_low"] = pd.cut(df["fico_range_low"], 20)
    df["income_to_installment"] = pd.qcut(df["income_to_installment"], 20)

    # print(weight_of_evidence(df, var_name='grade', good_bad_var='good_bad').to_string())
    # print(weight_of_evidence(df, var_name='home_ownership', good_bad_var='good_bad').to_string())
    # print(weight_of_evidence(df, var_name='addr_state', good_bad_var='good_bad').to_string())
    # print(weight_of_evidence(df, var_name='emp_length', good_bad_var='good_bad').to_string())

    plot_woe(weight_of_evidence(df, var_name='term', good_bad_var='good_bad'))
    plot_woe(weight_of_evidence(df, var_name='grade', good_bad_var='good_bad'))
    plot_woe(weight_of_evidence(df, var_name='sub_grade', good_bad_var='good_bad'))
    plot_woe(weight_of_evidence(df, var_name='home_ownership', good_bad_var='good_bad'))
    plot_woe(weight_of_evidence(df, var_name='addr_state', good_bad_var='good_bad'))
    plot_woe(weight_of_evidence(df, var_name='purpose', good_bad_var='good_bad'))
    plot_woe(weight_of_evidence(df, var_name='emp_length', good_bad_var='good_bad', discrete=False))
    plot_woe(weight_of_evidence(df, var_name='loan_amnt', good_bad_var='good_bad', discrete=False), rotation=35)
    plot_woe(weight_of_evidence(df, var_name='int_rate', good_bad_var='good_bad', discrete=False), rotation=35)
    plot_woe(weight_of_evidence(df, var_name='annual_inc', good_bad_var='good_bad', discrete=False), rotation=35)
    plot_woe(weight_of_evidence(df, var_name='dti', good_bad_var='good_bad', discrete=False), rotation=35)
    plot_woe(weight_of_evidence(df, var_name='fico_range_low', good_bad_var='good_bad', discrete=False), rotation=35)
    plot_woe(weight_of_evidence(df, var_name='income_to_installment', good_bad_var='good_bad', discrete=False), rotation=35)

    # # Print results for all loans and different subsets
    # print_test_results(f"CAGR on all loans:", df)
    # print_test_results(f"CAGR on odd loans:", df[df['loan_amnt_str'] != '000'])
    # print_test_results(f"CAGR on whole thousand loans:", df[df['loan_amnt_str'] == '000'])
    # print_test_results(f"CAGR on wedding loans:", df[df['purpose'] == 'wedding'])
    # print_test_results(f"CAGR on car loans:", df[df['purpose'] == 'car'])
    # print_test_results(f"CAGR on small business loans:", df[df['purpose'] == 'small_business'])
    # print_test_results(f"CAGR on home renters:", df[df['home_ownership'] == 'RENT'])
    # print_test_results(f"CAGR on home owners:", df[np.logical_or(df['home_ownership'] == 'OWN', df['home_ownership'] == 'MORTGAGE')])
    # print_test_results(f"CAGR on grade A loans:", df[df['grade'] == "A"])
    # print_test_results(f"CAGR on grade B loans:", df[df['grade'] == "B"])
    # print_test_results(f"CAGR on grade C loans:", df[df['grade'] == "C"])
    # print_test_results(f"CAGR on grade D loans:", df[df['grade'] == "D"])
    # print_test_results(f"CAGR on grade E loans:", df[df['grade'] == "E"])
    # print_test_results(f"CAGR on grade F loans:", df[df['grade'] == "F"])
    # print_test_results(f"CAGR on Wal-Mart loans:", df[df['emp_title'] == 'Wal-Mart'])
    # print_test_results(f"CAGR on US Army loans:", df[df['emp_title'] == 'US Army'])
    # print_test_results(f"CAGR on Wells Fargo loans:", df[df['emp_title'] == 'Wells Fargo'])
    # print_test_results(f"CAGR on income below 12.5k:", df[df['annual_inc'] < 12500])
    # print_test_results(f"CAGR on income above 12.5k and below 25k:", df[np.logical_and(df['annual_inc'] > 12500, df['annual_inc'] < 25000)])
    # print_test_results(f"CAGR on income above 25k and below 50k:", df[np.logical_and(df['annual_inc'] > 25000, df['annual_inc'] < 50000)])
    # print_test_results(f"CAGR on income above 50k and below 100k:", df[np.logical_and(df['annual_inc'] > 50000, df['annual_inc'] < 100000)])
    # print_test_results(f"CAGR on income above 100k and below 200k:", df[np.logical_and(df['annual_inc'] > 100000, df['annual_inc'] < 200000)])
    # print_test_results(f"CAGR on income above 200k and below 400k:", df[np.logical_and(df['annual_inc'] > 200000, df['annual_inc'] < 400000)])
    # print_test_results(f"CAGR on income above 400k:", df[df['annual_inc'] > 400000])
