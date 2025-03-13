import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kstest
from sklearn.preprocessing import LabelEncoder
from .correlations import compute_correlations


class BivariateEffectiveness:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None,
    ) ->  float:

        real_corr = compute_correlations(real_data, discrete_columns)
        synthetic_corr = compute_correlations(synthetic_data, discrete_columns)
        
        idx = np.tril_indices(n=len(real_corr.columns), k=-1)
        
        real_corr_values = real_corr.values[idx]
        synthetic_corr_values = synthetic_corr.values[idx]
        
        p_value = kstest(real_corr_values, synthetic_corr_values).pvalue
        
        return p_value