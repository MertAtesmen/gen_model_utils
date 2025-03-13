from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

from sklearn.preprocessing import OneHotEncoder

# TODO:
# - Make sure this class not only removes them but if specified caps the outlier values to their limits.
# - This class does not take categorical values into account so make sure you fix that.
# - Add an option to see how many outlier points are and their percentages in each variable

class OutlierHandler(TransformerMixin, BaseEstimator):
    """
    Remove outliers from the dataset
    
    This class derives from the sklearn classes and can be used like them.
    
    Outliers will be removed from the data acording to the lower and upper limits.
    
    Default limits:
    ------------------------------------
    - if the model is normally distrubuted:

    >>> lower_limit = np.mean(feature) - 3 * np.std(feature)
    >>> upper_limit = np.mean(feature) + 3 * np.std(feature)

    - if the model is skewed:
    
    >>> quantile_25 = np.quantile(feature, 0.25)
    >>> quantile_75 = np.quantile(feature, 0.75)
    
    >>> iqr = quantile_75 - quantile_25
    
    >>> lower_limit = quantile_25 - 1.5 * iqr
    >>> upper_limit = quantile_75 + 1.5 * iqr

    """
    
    def __init__(
        self, 
        limits: list[tuple[float, float]] | None = None,
        normal = False,
    ) -> None:
        super().__init__()
        self.limits = limits
        self.normal = normal

    def fit(self, X: np.ndarray, y=None):
        statistical_params = []
        
        for i in range(X.shape[1]):
            if self.limits and i < len(self.limits):
                statistical_params += [{
                    'lower_limit': self.limits[i][0],
                    'upper_limit': self.limits[i][1]
                    }]
            else:
                statistical_params += [self._calculate_statistical_params(X[:, i])]
        
        self._statistical_params = statistical_params
        
        return self
    
    def transform(self, X):
        for i in range(X.shape[1]):
            lower_limit = self._statistical_params[i]['lower_limit']
            upper_limit = self._statistical_params[i]['upper_limit']
            indices = (lower_limit < X[:, i]) & (X[:, i] < upper_limit)
            X = X[indices]
            
        return X
    
    def _calculate_statistical_params(self, feature: np.ndarray) -> dict[str, float]:
        
        if self.normal:
            lower_limit = np.mean(feature) - 3 * np.std(feature)
            upper_limit = np.mean(feature) + 3 * np.std(feature)
        else:
            quantile_25 = np.quantile(feature, 0.25)
            quantile_75 = np.quantile(feature, 0.75)
            
            iqr = quantile_75 - quantile_25
            
            lower_limit = quantile_25 - 1.5 * iqr
            upper_limit = quantile_75 + 1.5 * iqr

        return {
            'lower_limit': lower_limit,
            'upper_limit': upper_limit 
        }