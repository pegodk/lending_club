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

    # Remove all "current" loans and create new feature called "default"
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
    df['default'] = [1 if string == "Charged Off" else 0 for string in df['loan_status']]

    # Fill NA
    df['emp_length'].fillna('None', inplace=True)
    df['emp_title'].fillna('None', inplace=True)

    # Convert "term" to int number
    df['term'] = [int(str(string)[:3]) for string in df['term']]

    # Calculate return on investment as a percentage
    df['return_inv'] = df.eval("(total_pymnt / funded_amnt)**(1/(term / 12)) - 1") * 100

    # Save cleaned dataset to csv file
    df.to_csv('../data/processed/dataset_cleaned.csv', index=False)

    print(df[:100].to_string())
