import numpy as np
import pandas as pd

from typing import Literal

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from scipy.stats import pearsonr, pointbiserialr
from scipy.stats.contingency import association



def compute_correlations(
    X: pd.DataFrame,
    discrete_columns: list[str] = [],
    continuous_method: Literal["pearson"] = "pearson",
    nominal_method: Literal["cramer", "tschuprow", "pearson"] = "cramer",
    ignore_binary: bool = False,
) -> pd.DataFrame:

    continuous_columns = [col for col in X.columns if col not in discrete_columns]
    binary_columns = [col for col in discrete_columns if X[col].nunique() <= 2]
    nominal_columns = [col for col in discrete_columns if X[col].nunique() > 2]    
    
    correlations = pd.DataFrame(index=X.columns, columns=X.columns)

    continuous_set = set(continuous_columns)
    binary_set = set(binary_columns)
    nominal_set = set(nominal_columns)
    
    if ignore_binary:
        nominal_set = nominal_set.union(binary_set)
        binary_set = set()

    for i in range(len(X.columns)):
        correlations.iloc[i, i] = 1.0
        
    for col1, col2 in combinations(X.columns, r=2):
        if col1 in continuous_set and col2 in continuous_set:
            corr = pearsonr(X[col1], X[col2]).statistic
        elif col1 in nominal_set and col2 in nominal_set:
            frequency_table = pd.crosstab(X[col1], X[col2])
            corr = association(frequency_table, method=nominal_method)
        elif col1 in binary_set and col2 in binary_set:
            frequency_table = pd.crosstab(X[col1], X[col2])
            corr = phi_coefficient(frequency_table)
        elif col1 in binary_set and col2 in continuous_set:
            corr = pointbiserialr(X[col1].astype('category').cat.codes.astype('bool'), X[col2].astype('category').cat.codes).statistic 
        elif col1 in continuous_set and col2 in binary_set:
            corr = pointbiserialr(X[col2].astype('category').cat.codes.astype('bool'), X[col1].astype('category').cat.codes).statistic 
        elif col1 in nominal_set and col2 in continuous_set:
            corr = eta_coefficient(X, col2, col1)
        elif col1 in continuous_set and col2 in nominal_set:
            corr = eta_coefficient(X, col1, col2)    
        else:
            # Nominal and binary not defined in the paper
            frequency_table = pd.crosstab(X[col1], X[col2])
            corr = association(frequency_table, method=nominal_method)
            
        correlations.loc[col1, col2] = corr
        correlations.loc[col2, col1] = corr
        
    correlations = correlations.astype(float)
    
    correlations.fillna(1.0, inplace=True)
    
    return correlations


def phi_coefficient(
    f: pd.DataFrame
) -> float:
    if f.shape[0] < 2 or f.shape[1] < 2:
        return 1.0

    x = (f.iloc[0, 0] * f.iloc[1, 1] - f.iloc[1, 0] * f.iloc[0, 1])
    y = np.sqrt(
        f.iloc[0].sum() *  f.iloc[1].sum() * f.iloc[:, 0].sum()* f.iloc[:, 1].sum()
    )

    return x / y


def eta_coefficient(
    data: pd.DataFrame,
    c_col: str,
    n_col: str,
):
    overall_mean = data[c_col].mean()

    ss_total = ((data[c_col] - overall_mean) ** 2).sum()

    group_means = data.groupby(n_col)[c_col].mean()
    group_sizes = data.groupby(n_col)[c_col].size()
    ss_between = sum(group_sizes[g] * (group_means[g] - overall_mean)**2 for g in group_means.index)

    eta_squared_value = ss_between / ss_total
    return eta_squared_value

