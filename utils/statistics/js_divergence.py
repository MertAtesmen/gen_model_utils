import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon

class JSDivergence:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None,
        compute_continuous: bool = False,
    ) -> float:

        if not compute_continuous and len(discrete_columns) == 0:
            return 0

        if continuous_columns is None:
            continuous_columns = [col for col in real_data.columns if col not in discrete_columns]
        
        js_divergence_values = []

        for col in real_data.columns:
            real_data_col = real_data[col].dropna()
            synthetic_data_col = synthetic_data[col].dropna()

            concatenated_columns: pd.Series = pd.concat([real_data_col, synthetic_data_col])
            
            if col in discrete_columns:
                unique_values = concatenated_columns.unique()
                real_distrubition = real_data_col.value_counts(normalize=True).reindex(unique_values, fill_value = 0).values
                synthetic_distrubition = synthetic_data_col.value_counts(normalize=True).reindex(unique_values, fill_value = 0).values

            elif col in continuous_columns and compute_continuous:
                bin_edges = np.histogram_bin_edges(concatenated_columns, bins=10)

                counts_real, _ = np.histogram(real_data_col, bins=bin_edges, density=True)
                counts_synth, _ = np.histogram(synthetic_data_col, bins=bin_edges, density=True)

                real_distrubition = counts_real / counts_real.sum()
                synthetic_distrubition = counts_synth / counts_synth.sum()

            else:
                continue
            
            js_divergence_values.append(jensenshannon(real_distrubition, synthetic_distrubition))

        return np.mean(js_divergence_values)
