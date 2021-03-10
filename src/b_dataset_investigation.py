import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from src.utils import print_test_results, calc_annual_return_vec


if __name__ == "__main__":
    # Read the datasets
    df = pd.read_csv('../data/processed/dataset_cleaned.csv')

    # New feature of whether loan_amnt is a whole thousand or odd
    df["loan_amnt_str"] = [str(int(amnt))[-3:] for amnt in df["loan_amnt"]]

    v1 = df[df['loan_amnt_str'] != '000']['default']
    v2 = df[df['loan_amnt_str'] == '000']['default']
    print(ttest_ind(v1, v2))

    # Print results for all loans and different subsets
    print_test_results(f"CAGR on all loans:", df)
    print_test_results(f"CAGR on odd loans:", df[df['loan_amnt_str'] != '000'])
    print_test_results(f"CAGR on whole thousand loans:", df[df['loan_amnt_str'] == '000'])
    print_test_results(f"CAGR on wedding loans:", df[df['purpose'] == 'wedding'])
    print_test_results(f"CAGR on car loans:", df[df['purpose'] == 'car'])
    print_test_results(f"CAGR on small business loans:", df[df['purpose'] == 'small_business'])
    print_test_results(f"CAGR on home renters:", df[df['home_ownership'] == 'RENT'])
    print_test_results(f"CAGR on home owners:", df[np.logical_or(df['home_ownership'] == 'OWN', df['home_ownership'] == 'MORTGAGE')])
    print_test_results(f"CAGR on grade A loans:", df[df['grade'] == "A"])
    print_test_results(f"CAGR on grade B loans:", df[df['grade'] == "B"])
    print_test_results(f"CAGR on grade C loans:", df[df['grade'] == "C"])
    print_test_results(f"CAGR on grade D loans:", df[df['grade'] == "D"])
    print_test_results(f"CAGR on grade E loans:", df[df['grade'] == "E"])
    print_test_results(f"CAGR on grade F loans:", df[df['grade'] == "F"])
    print_test_results(f"CAGR on Wal-Mart loans:", df[df['emp_title'] == 'Wal-Mart'])
    print_test_results(f"CAGR on US Army loans:", df[df['emp_title'] == 'US Army'])
    print_test_results(f"CAGR on Wells Fargo loans:", df[df['emp_title'] == 'Wells Fargo'])
    print_test_results(f"CAGR on income below 12.5k:", df[df['annual_inc'] < 12500])
    print_test_results(f"CAGR on income above 12.5k and below 25k:", df[np.logical_and(df['annual_inc'] > 12500, df['annual_inc'] < 25000)])
    print_test_results(f"CAGR on income above 25k and below 50k:", df[np.logical_and(df['annual_inc'] > 25000, df['annual_inc'] < 50000)])
    print_test_results(f"CAGR on income above 50k and below 100k:", df[np.logical_and(df['annual_inc'] > 50000, df['annual_inc'] < 100000)])
    print_test_results(f"CAGR on income above 100k and below 200k:", df[np.logical_and(df['annual_inc'] > 100000, df['annual_inc'] < 200000)])
    print_test_results(f"CAGR on income above 200k and below 400k:", df[np.logical_and(df['annual_inc'] > 200000, df['annual_inc'] < 400000)])
    print_test_results(f"CAGR on income above 400k:", df[df['annual_inc'] > 400000])
