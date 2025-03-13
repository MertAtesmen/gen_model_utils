import pandas as pd
import numpy as np
from typing import Literal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, RobustScaler, MinMaxScaler

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        discrete_columns: list[str] = [],
        scaler: Literal['robust', 'minmax', 'standart'] = 'standart',
        encoder: Literal['one_hot', 'ordinal'] = 'one_hot'
    ):
        super().__init__()
        self._discrete_columns = set(discrete_columns)
        self._scaler = scaler
        self._encoder = encoder

    def get_metdata(self, X:pd.DataFrame):
        meta_data = []

        for col in X.columns:
            if col in self._discrete_columns:
                if self._encoder == 'one_hot':
                    meta_data.append({
                        'name': col,
                        'type': 'discrete',
                        'len': len(X[col].unique())
                    })
                elif self._encoder == 'ordinal':
                    meta_data.append({
                        'name': col,
                        'type': 'discrete',
                        'len': 1
                    })
                else: 
                    raise ValueError(f'No encoder of the type ({self._encoder}) is supported')
            else:
                meta_data.append({
                    'name': col,
                    'type': 'numeric',
                })
                
        return meta_data
    
    def _get_encoder(self):
        if self._encoder == 'one_hot':
            return OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif self._encoder == 'ordinal':
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            raise ValueError(f'No encoder of the type ({self._encoder}) is supported') 
        
    def _get_scaler(self):
        if self._scaler == 'robust':
            return RobustScaler()
        elif self._scaler == 'minmax':
            return MinMaxScaler()
        elif self._scaler == 'standart':
            return StandardScaler()
        else:
            raise ValueError(f'No scaler of the type ({self._scaler}) is supported') 
    
    
    def fit(self, X: pd.DataFrame, y=None):
        transformers = []
        
        self._metadata = self.get_metdata(X)
        
        for idx, col in enumerate(X.columns):
            column_values = X.loc[:, col].to_frame()
            
            if self._metadata[idx]['type'] == 'discrete':
                ohe = self._get_encoder()
                ohe.set_output(transform='pandas')
                ohe.fit(column_values)
                transformers.append(ohe)
            else:
                scaler = self._get_scaler()
                scaler.set_output(transform='pandas')
                scaler.fit(column_values)
                transformers.append(scaler)

        self._transformers = transformers        
        
        return self
    
    def transform(self, X: pd.DataFrame):
        col_converted = []
        
        for idx, col in enumerate(X.columns):
            column_values = X.loc[:, col].to_frame()
            
            col_converted.append(self._transformers[idx].transform(column_values))
            
        return pd.concat(col_converted, axis=1)
    
    def inverse_transform(self, X: pd.DataFrame):
        col_inversed = []
        
        current_idx = 0
        for idx in range(len(self._metadata)):
            if self._metadata[idx]['type'] == 'discrete':
                l = self._metadata[idx]['len']
                column_values = X.iloc[:, current_idx: current_idx + l]
                current_idx += l
            else:
                column_values = X.iloc[:, current_idx: current_idx + 1]
                current_idx += 1

            col_inversed.append(pd.Series(self._transformers[idx].inverse_transform(column_values).reshape(-1), name=self._metadata[idx]['name']))

        return pd.concat(col_inversed, axis=1)
