"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from ._ctabgan_plus.data_preparation import DataPrep
from ._ctabgan_plus.ctabganplus_synthesizer import CTABGANPlusSynthesizer
import torch

import warnings

warnings.filterwarnings("ignore")

from typing import TypeVar, Type

T = TypeVar('T', bound='CTABGANPlus')

class CTABGANPlus():

    def __init__(self,
                 df,
                 categorical_columns = [],
                 log_columns = [],
                 mixed_columns= {},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type= {},
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=300,
                 lr=2e-4,
                 device="cuda"):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANPlusSynthesizer(
                class_dim=class_dim,
                random_dim=random_dim,
                num_channels=num_channels,
                l2scale=l2scale,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs,
                device=device
        )
        self.raw_df = df
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
                
    def fit(self):
        
        start_time = time.time()
        # Change this
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, num_samples, seed=0):
        
        sample = self.synthesizer.sample(num_samples, seed) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
    
    def save(self, path: str):
        torch.save(self, path)
    
    @classmethod
    def load(cls, path) -> T:
        return torch.load(path)      