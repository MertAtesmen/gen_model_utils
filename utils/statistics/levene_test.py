import numpy as np
import pandas as pd

from scipy.stats import levene

from typing import Literal

class LeveneTest:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None,
        return_options: Literal['statistic', 'p_value', 'all'] = 'p_value'

    ) -> tuple[float, float, dict]:

        if return_options not in ('statistic', 'p_value', 'all'):
            raise ValueError(f"return_options can not take the value {return_options} the valid argumnets are: ('statistic', 'p_value', 'all')")

        if continuous_columns is None:
            continuous_columns = [col for col in real_data.columns if col not in discrete_columns]    
            
        if len(continuous_columns) == 0:
            if return_options in ('p_value', 'statistic'):
                return 0
            else:
                return 0, 0, dict()

        statistic_values = []
        p_values = []
        
        for col in continuous_columns:

            real_data_col = real_data[col].dropna()
            synthetic_data_col = synthetic_data[col].dropna()
        
            statistic, p_value = levene(real_data_col, synthetic_data_col)
            
            if np.isnan(p_value):
                p_value = 1.0
            
            statistic_values.append(statistic)
            p_values.append(p_value)
        
        col_values = {
            continuous_columns[idx]: {'statistic': statistic_values[idx], 'p_value': p_values[idx]} 
            for idx in range(len(continuous_columns))
        }
        
        if return_options == 'p_value':
            return np.mean(p_values)
        elif return_options == 'statistic':
            return np.mean(statistic_values)
        else:
            return np.mean(statistic_values), np.mean(p_values), col_values