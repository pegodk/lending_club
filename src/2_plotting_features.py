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
from src.utils import *

# Read the datasets
df = pd.read_csv('../data/processed/dataset_cleaned.csv')

print("The data has {0} rows and {1} fields".format(*df.shape))

# df = df[df.loan_status.isin(['Fully Paid', 'Charged Off'])]
# df['int_rate'] = df['int_rate'].apply(process_int_rate)
# df['term'] = df['term'].apply(process_term)
# df['grade2'] = df['grade'].apply(process_grade)
# df['emp_length'] = df['emp_length'].apply(process_emp_length)
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


plt.figure()
df.default.value_counts().plot(kind='bar')
plt.tight_layout()
plt.savefig('../results/plots/Histogram___FullyPaid.png')

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

bin_idx = np.linspace(0, 0.25, 40)
def_rates_by_hist(df, 'installment_over_income', bin_idx=bin_idx)
