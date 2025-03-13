from typing import Literal

import numpy as np
import pandas as pd

from .kolmogorov_smirnov_test import KolmogorovSmirnovTest
from .mann_whitney_u_test import MannWhitneyUTest
from .levene_test import LeveneTest
from .chi_squared_test import ChiSquaredTest
from .wasserstein_distance import WassersteinDistance
from .bivariate_effectiveness import BivariateEffectiveness

from .js_divergence import JSDivergence
from .pairwise_correlation_difference import PairwiseCorrelationDifference
from .propensity_score import PropensityScore

from .pca_metric import PCAMetric

from ..dataset_utils import convert_ctabgan_data

from copy import deepcopy


class StatisticalSimilarity:
    """Computes the statistical similarity between two datasets or multiple tuples then returns the results and aggreagates if possible."""

    def __init__(
        self,
        discrete_columns: list[str],
        verbose = 0,
        **kwargs,
    ):
        self.discrete_columns = discrete_columns
        self.metrics = kwargs
        self._verbose = verbose

        pass
    
    def calculate_statistical_similarity(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ):
        statistical_similarities = {}

        if self._verbose > 0:
            print(f'{"START":<10}: Calculating The STATISTICAL SIMILARITY')

        synthetic_data = convert_ctabgan_data(real_data, synthetic_data)
    
        # self.metrics is a dictionary
        statistical_similarities.update(
            {metric: self._calculate_metric(real_data, synthetic_data, metric, self.metrics[metric]) for metric in self.metrics}
        )

        return statistical_similarities
        

    def calculate_statistical_similarity_n(
        self,
        real_data: list[pd.DataFrame],
        synthetic_data: list[pd.DataFrame],
    ):
        statistical_similarities = {}

        for idx in range(len(real_data)):
            statistical_similarities[f'fold{idx + 1}'] = self.calculate_statistical_similarity(real_data[idx], synthetic_data[idx])

        score_values = {key: [] for key in statistical_similarities['fold1'].keys()}
        for fold_name in statistical_similarities.keys():
            for score_name in score_values.keys():
                score_values[score_name].append(statistical_similarities[fold_name][score_name])

        statistical_similarities['Average'] = {score_name: np.mean(score_values[score_name]) for score_name in score_values}

        return statistical_similarities
        
    def _calculate_metric(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metric: str,
        metric_args: dict | list
    ):
        _metric_args = deepcopy(metric_args)
        
        if 'name' in _metric_args:
            metric = _metric_args['name']
            del _metric_args['name']
            
        if self._verbose > 2:
            print(f'Calculating the Metric: {metric}')
        
        if metric == 'KS':
            return KolmogorovSmirnovTest.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'MW':
            return MannWhitneyUTest.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'LE':
            return LeveneTest.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'CT':
            return ChiSquaredTest.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'BE':
            return BivariateEffectiveness.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'MMD':
            return np.nan
        elif metric == 'WD':
            return WassersteinDistance.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'JSD':
            return JSDivergence.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'pMSE':
            return PropensityScore.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'PCD':
            return PairwiseCorrelationDifference.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        elif metric == 'PCA':
            return PCAMetric.compute(real_data, synthetic_data, self.discrete_columns, **_metric_args)
        else:
            raise ValueError(f'No metric named ({metric}) exists')
