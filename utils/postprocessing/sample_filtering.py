import pandas as pd
import numpy as np
from typing import Literal

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from ..preprocessing.data_transformer import DataTransformer


# TODO: Add filtering options. i.e: `majority_only`, `all`, `not_minority`.
class SampleFiltering:
    def __init__(
        self,
        discrete_columns: list[str] = [],
    ):
        super().__init__()
        self._discrete_columns = discrete_columns
        
    def resample(
        self,
        real_X: pd.DataFrame,
        real_y: pd.Series | np.ndarray,
        synthetic_X: pd.DataFrame,
        synthetic_y: pd.Series | np.ndarray,
    ):
        label_encoder = LabelEncoder()
        label_encoder.fit(np.concatenate((real_y, synthetic_y)))      

        y_train = label_encoder.transform(real_y)
        y_test = label_encoder.transform(synthetic_y)        
        
        transformer = DataTransformer(discrete_columns=self._discrete_columns, encoder='ordinal')
        X_train = transformer.fit_transform(real_X)
        X_test = transformer.transform(synthetic_X)

        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        resampled_idx = y_test == y_pred
        
        return synthetic_X[resampled_idx], synthetic_y[resampled_idx]