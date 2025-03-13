import pandas as pd
import numpy as np

from .correlations import compute_correlations

# Minimize
class PairwiseCorrelationDifference:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None,
        ignore_binary: bool = False,
    ) -> float:

        if continuous_columns is None:
            continuous_columns = [col for col in real_data.columns if col not in discrete_columns]

        real_data_corr = compute_correlations(real_data, discrete_columns=discrete_columns, ignore_binary=ignore_binary)
        synthetic_data_corr = compute_correlations(synthetic_data,discrete_columns=discrete_columns, ignore_binary=ignore_binary)

        pcorr_difference = np.linalg.norm(real_data_corr - synthetic_data_corr)

        return pcorr_difference