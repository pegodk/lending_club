import numpy as np
import pandas as pd
from src.utils import print_test_results

if __name__ == "__main__":
    # Read the datasets
    df = pd.read_csv('../data/processed/dataset_cleaned.csv')

    print("The data has {0} rows and {1} fields".format(*df.shape))

    print_test_results(f"CAGR on all loans:", df)

    idx = df['purpose'] == 'wedding'
    print_test_results(f"CAGR on wedding loans:", df[idx])

    idx = df['purpose'] == 'car'
    print_test_results(f"CAGR on car loans:", df[idx])

    idx = df['purpose'] == 'small_business'
    print_test_results(f"CAGR on small business loans:", df[idx])

    idx = df['home_ownership'] == 'RENT'
    print_test_results(f"CAGR on home renters:", df[idx])

    idx = np.logical_or(df['home_ownership'] == 'OWN', df['home_ownership'] == 'MORTGAGE')
    print_test_results(f"CAGR on home owners:", df[idx])

    idx = df['grade'] == "A"
    print_test_results(f"CAGR on grade A loans:", df[idx])

    idx = df['grade'] == "B"
    print_test_results(f"CAGR on grade B loans:", df[idx])

    idx = df['grade'] == "C"
    print_test_results(f"CAGR on grade C loans:", df[idx])

    idx = df['grade'] == "D"
    print_test_results(f"CAGR on grade D loans:", df[idx])

    idx = df['grade'] == "E"
    print_test_results(f"CAGR on grade E loans:", df[idx])

    idx = df['grade'] == "F"
    print_test_results(f"CAGR on grade F loans:", df[idx])

    idx = df['emp_title'] == 'Wal-Mart'
    print_test_results(f"CAGR on Wal-Mart loans:", df[idx])

    idx = df['emp_title'] == 'US Army'
    print_test_results(f"CAGR on US Army loans:", df[idx])

    idx = df['emp_title'] == 'Wells Fargo'
    print_test_results(f"CAGR on Wells Fargo loans:", df[idx])

    idx = df['annual_inc'] < 12500
    print_test_results(f"CAGR on income below 12.5k:", df[idx])

    idx = np.logical_and(df['annual_inc'] > 12500, df['annual_inc'] < 25000)
    print_test_results(f"CAGR on income above 12.5k and below 25k:", df[idx])

    idx = np.logical_and(df['annual_inc'] > 25000, df['annual_inc'] < 50000)
    print_test_results(f"CAGR on income above 25k and below 50k:", df[idx])

    idx = np.logical_and(df['annual_inc'] > 50000, df['annual_inc'] < 100000)
    print_test_results(f"CAGR on income above 50k and below 100k:", df[idx])

    idx = np.logical_and(df['annual_inc'] > 100000, df['annual_inc'] < 200000)
    print_test_results(f"CAGR on income above 100k and below 200k:", df[idx])

    idx = np.logical_and(df['annual_inc'] > 200000, df['annual_inc'] < 400000)
    print_test_results(f"CAGR on income above 200k and below 400k:", df[idx])

    idx = df['annual_inc'] > 400000
    print_test_results(f"CAGR on income above 400k:", df[idx])
