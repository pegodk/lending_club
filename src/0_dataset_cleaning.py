import numpy as np
import pandas as pd

if __name__ == "__main__":

    columns_subset = ["loan_amnt",
                      "funded_amnt",
                      "term",
                      "int_rate",
                      "installment",
                      "grade",
                      "sub_grade",
                      "emp_title",
                      "emp_length",
                      "home_ownership",
                      "annual_inc",
                      "issue_d",
                      "loan_status",
                      "purpose",
                      "title",
                      "addr_state",
                      "dti",
                      "earliest_cr_line",
                      "fico_range_low",
                      "fico_range_high",
                      "total_acc",
                      "total_pymnt"]

    df = pd.read_csv('../data/raw/accepted_2007_to_2018Q4.csv', usecols=columns_subset)
    # df = pd.read_csv('../data/raw/accepted_2007_to_2018Q4.csv', usecols=columns_subset, nrows=10000)

    # Remove all "current" loans and create new feature called "default"
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
    df['default'] = [1 if string == "Charged Off" else 0 for string in df['loan_status']]
    # df.drop(columns="loan_status", inplace=True)

    # # Clean employment length: -1 for NaN, 0 for <1 year and 10 for 10+
    # df['emp_length'].fillna(-1, inplace=True)
    # df['emp_length'].replace("< 1 year", 0, inplace=True)
    # df['emp_length'] = [int(str(string).split(" ")[0].split("+")[0]) for string in df['emp_length']]

    # Convert "term" to int number
    df['term'] = [int(str(string)[:3]) for string in df['term']]

    # Calculate return on investment as a percentage
    df['return_inv'] = df.eval("(total_pymnt / funded_amnt)**(1/(term / 12)) - 1") * 100

    # # Convert "sub_grade" score to integers from 0 (A1) to 25 (C5)
    # subgrade_sorted = sorted(np.unique(df["sub_grade"]))
    # df['sub_grade'] = [subgrade_sorted.index(subgrade) for subgrade in df['sub_grade']]

    # Save cleaned dataset to csv file
    df.to_csv('../data/processed/dataset_cleaned.csv', index=False)

    print(df[:100].to_string())
