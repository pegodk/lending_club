import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import process_emp_length, process_home_ownership, convert_to_num

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
                      "total_pymnt"]

    df = pd.read_csv('../data/raw/accepted_2007_to_2018Q4.csv', usecols=columns_subset)

    # Create new features
    df['installment_to_income'] = df['installment'] / (df['annual_inc'] / 12)

    # Remove all "current" loans and create new feature called "default"
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]
    df['good_bad'] = [1 if string == "Fully Paid" else 0 for string in df['loan_status']]

    # Fill NA and convert emp_length to int number
    df['emp_title'].fillna('None', inplace=True)
    df['emp_length'].fillna('None', inplace=True)
    df['emp_length'] = df['emp_length'].apply(process_emp_length)

    # Put ANY/OTHER/NONE together with RENT in home_ownership
    df['home_ownership'] = df['home_ownership'].apply(process_home_ownership)

    # Convert "term" to int number
    df['term'] = [int(str(string)[:3]) for string in df['term']]

    # Calculate return on investment as a percentage
    df['return_inv'] = df.eval("(total_pymnt / funded_amnt)**(1/(term / 12)) - 1") * 100

    # Split dataset into "train" and "test"
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Save cleaned dataset to csv file
    train.to_csv('../data/temp/dataset_train.csv', index=False)
    test.to_csv('../data/temp/dataset_test.csv', index=False)

    print(train[:10].to_string())
    print(test[:10].to_string())
