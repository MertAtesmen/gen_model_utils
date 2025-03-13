import pandas as pd
import numpy as np

from typing import Optional, Tuple


from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from typing import Literal



# Returns variance difference, angle difference, (real_proj, synthetic_proj)
class PCAMetric:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: Optional[list[str]]  = None,
        n_components: Optional[int] = None,
        only_continuous: bool = False,
        return_options: Literal['angle', 'variance', 'all'] = 'variance'
    ) -> Tuple[float, float, Tuple[pd.DataFrame, pd.DataFrame]]:

        if return_options not in ('angle', 'variance', 'all'):
            raise ValueError(f"return_options can not take the value {return_options} the valid argumnets are: ('angle', 'variance', 'all')")

        if continuous_columns is None:
            continuous_columns = [col for col in real_data.columns if col not in discrete_columns]
            
        real_data, synthetic_data = _convert_datasets(real_data, synthetic_data, discrete_columns, only_continuous)
        
        if n_components is None:
            n_components = real_data.shape[1]
            
        if n_components > real_data.shape[1]:
            raise ValueError('The n_components is bigger than the feature size')
        
        factor = real_data.shape[1] / (real_data.shape[1] + n_components - 2)
        
        pca_real = PCA(n_components=n_components)
        pca_synthetic = PCA(n_components=n_components)
        
        pca_real.set_output(transform='pandas')
        pca_synthetic.set_output(transform='pandas')
        
        real_projected = pca_real.fit_transform(real_data)
        synthetic_projected = pca_synthetic.fit_transform(synthetic_data)
        
        var_difference = factor * np.sum(np.abs(pca_real.explained_variance_ratio_- pca_synthetic.explained_variance_ratio_))
        angle_diff = min([np.arccos(_bound(pca_real.components_[0] @ (s*pca_synthetic.components_[0]),(-1,1))) for s in [1,-1]]) * 2 / np.pi

        if return_options == 'variance':
            return var_difference
        elif return_options == 'angle':
            return angle_diff
        else:
            return var_difference, angle_diff, (real_projected, synthetic_projected)
    
    
def _bound(value: float, range: Tuple[float, float]) -> float:
    """Bound a value between two values
    
    Args:
        range (Tuple[float, float]): The lower and upper bounds
        value (float): The value to bound
    
    Returns:
        float: The bounded value
    
    Example:
    >>> _bound(0.5,(0, 1))
    0.5
    """
    low, high = range
    return max(low, min(high, value))

def _convert_datasets(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    discrete_columns: list[str] = [],
    only_continuous: bool = False,
):
    continuous_columns = [col for col in real_data.columns if col not in discrete_columns]
    
    real_data_transformed = real_data.copy(deep=True)
    synthetic_data_transformed = synthetic_data.copy(deep=True)
    
    if only_continuous:
        real_data_transformed.drop(columns=discrete_columns, inplace=True)
        synthetic_data_transformed.drop(columns=discrete_columns, inplace=True)
    else:
        transformer = ColumnTransformer(
            transformers=[
                ('discrete', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), discrete_columns)
            ],
            sparse_threshold=0,
            remainder='passthrough',
            verbose_feature_names_out=False
        )
    
        transformer.set_output(transform='pandas')

        real_data_transformed = transformer.fit_transform(real_data_transformed)
        synthetic_data_transformed = transformer.transform(synthetic_data_transformed)
    
    real_data_transformed[continuous_columns] = StandardScaler().fit_transform(real_data_transformed[continuous_columns])
    synthetic_data_transformed[continuous_columns] = StandardScaler().fit_transform(synthetic_data_transformed[continuous_columns])

    return real_data_transformed, synthetic_data_transformed