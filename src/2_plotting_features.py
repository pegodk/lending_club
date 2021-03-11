import numpy as np
import pandas as pd
from src.utils import def_rates_by_categorical, int_rates_by_categorical, returns_by_categorical, def_rates_by_hist

if __name__ == "__main__":
    # Read the datasets
    df = pd.read_csv('../data/processed/dataset_cleaned.csv')

    # Calculate "issue year"
    df['issue_year'] = df['issue_d'].apply(lambda x: int(x.split('-')[1]))

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
