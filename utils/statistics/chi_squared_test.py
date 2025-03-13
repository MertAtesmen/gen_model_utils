import numpy as np
import pandas as pd

from scipy.stats import chisquare

from typing import Literal


class ChiSquaredTest:
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
        
        for col in discrete_columns:
            
            real_data_col = real_data[col].dropna()
            synthetic_data_col = synthetic_data[col].dropna()

            concatenated_columns: pd.Series = pd.concat([real_data_col, synthetic_data_col])
            unique_values = concatenated_columns.unique()

            real_distrubition = real_data_col.value_counts(normalize=True).reindex(unique_values, fill_value = 0).values
            synthetic_distrubition = synthetic_data_col.value_counts(normalize=True).reindex(unique_values, fill_value = 0).values

            # Prevent NaN    
            real_distrubition += +1e-6
            real_distrubition /= np.sum(real_distrubition)
            synthetic_distrubition += +1e-6
            synthetic_distrubition /= np.sum(synthetic_distrubition)

            statistic, p_value = chisquare(real_distrubition, synthetic_distrubition)

            if np.isnan(p_value):
                p_value = 1.0

            statistic_values.append(statistic)
            p_values.append(p_value)
        
        
        col_values = {
            discrete_columns[idx]: {'statistic': statistic_values[idx], 'p_value': p_values[idx]} 
            for idx in range(len(discrete_columns))
        }
        
        if return_options == 'p_value':
            return np.mean(p_values)
        elif return_options == 'statistic':
            return np.mean(statistic_values)
        else:
            return np.mean(statistic_values), np.mean(p_values), col_values