"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from ._ctabgan.data_preparation import DataPrep
from ._ctabgan.ctabgan_synthesizer import CTABGANSynthesizer

import warnings
import torch

from typing import TypeVar, Type

T = TypeVar('T', bound='CTABGAN')

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(
        self,
        df,
        test_ratio = 0.0,
        categorical_columns = [], 
        log_columns = [],
        mixed_columns= {},
        integer_columns = [],
        problem_type= {},
        batch_size = 512,
        class_dim = (256, 256, 256, 256),
        random_dim: int = 128,
        num_channels: int = 64,
        l2scale: float = 0.00001,
        lr = 2e-4,
        epochs = 300,
        device=None
    ):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(lr = lr, epochs = epochs, batch_size = batch_size, class_dim = class_dim, device = device, random_dim=random_dim, l2scale=l2scale, num_channels=num_channels)
        self.raw_df = df
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        
    def fit(self, no_train=False):
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], 
        mixed = self.data_prep.column_types["mixed"],type=self.problem_type, no_train=no_train)


    def generate_samples(self, num_samples, seed=0):
        
        sample = self.synthesizer.sample(num_samples, seed) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
    
    def save(self, path: str):
        torch.save(self, path)
    
    @classmethod
    def load(cls, path) -> T:
        return torch.load(path)        
