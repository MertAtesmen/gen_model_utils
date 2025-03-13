import pandas as pd
import numpy as np
from typing import Union, Literal
from copy import deepcopy
import time

from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor


from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from ..preprocessing.data_transformer import DataTransformer
from ..dataset_utils import convert_dataset

class MLPerformance:
    """Computes the ml performance of multiple models and returns the aggregates and one by one results.

    Parameters
    ---
    discrete_columns : list[str]
        Discrete columns in the dataset.
        
    models : list[str], default = ['KNN', 'MLP', 'SVM', 'RF', 'Linear', 'LGBM'].
        Models the MLPerformance class uses to calculate performances
        `Linear` model is the LinearRegression model if the task is regression else it is the LogisticRegression model
    
    task : {'regression', 'classification', 'classification_binary'}, default='classification_binary'
        ML task for the dataset
        
    scaler : {'robust', 'minmax', 'standart'}, default='robust'
        Scaling method to use for the continuous columns. Every column othe than discrete columns     
        
    ordinal_only : bool, default=False
        This specifies if only the ordinal encoding method will be used for the dataset. 
        On the default scenario, svm, mlp, knn and linear models use one hot encoding
        while ordinal encoding is used for the other models.
        
    fast_computation : bool, default=False
        If False then the default computation steps are followed. If True then some of 
        the slower models will be skipped to get a fast computation
    """
        
    def __init__(
        self,
        discrete_columns: list[str],
        models: list[str] = ['KNN', 'MLP', 'SVM', 'RF', 'Linear', 'LGBM'],
        task: Literal['regression', 'classification', 'classification_binary'] = 'classification_binary',
        scaler: Literal['robust', 'minmax', 'standart'] = 'robust',
        score_metrics: list[str] = ['F1', 'AUCROC', 'AUCPR'],
        ordinal_only: bool = False,
        fast_computation: bool = False,
        verbose = 0,
    ):
        
        self.discrete_columns = discrete_columns
        self.models = models
        self.task = task
        self.scaler = scaler
        self.score_metrics = score_metrics
        self._ordinal_only = ordinal_only
        self._fast_computation = fast_computation
        self._verbose = verbose

        pass
    
    def calculate_ml_utility(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        ml_utilities = {}
        
        if self._verbose > 0:
            print(f'{"START":<10}: Calculating The ML Utility')
        
        X_train = convert_dataset(X_train)
        X_test = convert_dataset(X_test)
        y_train = convert_dataset(y_train)
        y_test = convert_dataset(y_test)    
        
        if self.task == 'classification_binary':
            _, counts = np.unique(y_test, return_counts=True)
            self._minority_class_idx = counts.argmin()

        
        if self.task != 'regression':
            le = LabelEncoder()
            le.fit(np.concatenate((y_train, y_test)))      

            y_train = le.transform(y_train)
            y_test = le.transform(y_test)
        
        ohe_models, ordinal_models = self._get_models()
        # Calculate the values
        
        ohe_transformer = DataTransformer(discrete_columns=self.discrete_columns)
        X_train_ohe = ohe_transformer.fit_transform(X_train)
        X_test_ohe = ohe_transformer.transform(X_test)
        
        ml_utilities.update(
            {str(model): self._train_model_and_get_scores(model, X_train_ohe, X_test_ohe, y_train, y_test) for model in ohe_models}
        )
        
        ordinal_transformer = DataTransformer(discrete_columns=self.discrete_columns, encoder='ordinal')
        X_train_ordinal = ordinal_transformer.fit_transform(X_train)
        X_test_ordinal = ordinal_transformer.transform(X_test)
        
        ml_utilities.update(
             {str(model): self._train_model_and_get_scores(model, X_train_ordinal, X_test_ordinal, y_train, y_test) for model in ordinal_models}
        )
        
        ml_utilities.update(
            self._get_aggregates(ml_utilities)
        )
        
        return ml_utilities
        

    def calculate_ml_utility_n(
        self,
        X_train: list[pd.DataFrame],
        X_test: list[pd.DataFrame],
        y_train: list[pd.DataFrame],
        y_test: list[pd.DataFrame],
    ):
        ml_utilites = {}
        _util_temp = {}
        for idx in range(len(X_train)):
            _util_temp[f'fold{idx + 1}'] = self.calculate_ml_utility(X_train[idx], X_test[idx], y_train[idx], y_test[idx])
        
        ml_utilites['details'] = _util_temp
        
        for fold_name in _util_temp.keys():
            ml_utilites[fold_name] = _util_temp[fold_name]['Average']

        score_values = {key: [] for key in ml_utilites['fold1'].keys()}
        for fold_name in _util_temp.keys():
            for score_name in score_values.keys():
                score_values[score_name].append(ml_utilites[fold_name][score_name])
                
        ml_utilites['Average'] = {score_name: np.mean(score_values[score_name]) for score_name in score_values}

        return ml_utilites
        
    def _get_models(self):
        ohe_models = []
        ordinal_models = []
        
        for model in self.models:
            if model == 'KNN':
                ohe_models.append(KNeighborsClassifier() if self.task != 'regression' else KNeighborsRegressor())
            elif model == 'MLP' and not self._fast_computation:
                ohe_models.append(MLPClassifier() if self.task != 'regression' else MLPRegressor())
            elif model == 'SVM' and not self._fast_computation:
                ohe_models.append(SVC(probability=True, max_iter=1000) if self.task != 'regression' else SVR(probability=True, max_iter=1000))
            elif model == 'RF':
                ordinal_models.append(RandomForestClassifier() if self.task != 'regression' else RandomForestRegressor())
            elif model == 'Linear':
                ohe_models.append(LogisticRegression() if self.task != 'regression' else LinearRegression())
            elif model == 'LGBM':
                ordinal_models.append(LGBMClassifier(verbose=-1) if self.task != 'regression' else LGBMRegressor(verbose=-1))
            else:
                raise ValueError(f'No model named ({model}) exists')
            
        return ohe_models, ordinal_models
    
    def _train_model_and_get_scores(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ):
        if self._verbose > 2:
            print(f'{"TRAIN":<10}: Training the {model} model')
            
        start_time = time.perf_counter()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if self.task != 'regression':
            predict_probas = model.predict_proba(X_test)

        if self.task == 'classification_binary':
            if predict_probas.shape[1] == 1:
                predict_probas = np.zeros_like(predictions)
            else:
                predict_probas = predict_probas[:, 1]

        end_time = time.perf_counter()

        _iteration_string = "" if not hasattr(model, 'n_iter_') else f'{model.n_iter_} iteration and '
        
        if self._verbose > 2:
            print(f'{"TRAIN":<10}: Training finished in {_iteration_string}{(end_time - start_time) * 1000:.3f} ms')
            
        
        scores = {}
        
        for metric in self.score_metrics:
            scores[metric] = self._calculate_score_metric(metric, y_test, predictions, predict_probas) 
        
        return scores
    
    # TODO: Add more metrics and see how the AUC is calculated for the binary
    def _calculate_score_metric(
        self, 
        metric_name: str,
        y_true,
        predictions,
        predict_probas
    ):
        if metric_name == 'F1':
            return f1_score(y_true, predictions, average='macro') if self.task == 'classifcation' else \
                f1_score(y_true, predictions, average='binary', pos_label=self._minority_class_idx)
        elif metric_name == 'AUCROC':
            return roc_auc_score(y_true, predict_probas, average='macro')
        elif metric_name == 'AUCPR':    
            return average_precision_score(y_true, predict_probas, average='macro')


    def _get_aggregates(self, scores: dict):    
        score_names = list(
            list(scores.values())[0]
        )
        
        all_scores = {key: [] for key in score_names}

        for model_name in scores.keys():
            for score_name in score_names:
                all_scores[score_name].append(scores[model_name][score_name])

        # TODO: Add more aggregate if possible? 
        aggregates = {} 
        aggregates['Average'] = {key: np.mean(all_scores[key]) for key in all_scores.keys()}
            
        return aggregates
    

# OLD ML UTILITY FUNCTION

def calculate_ml_utility(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray] = None,
    y_test: Union[pd.Series, np.ndarray] = None,
    discrete_columns: list[str] = [],
    classification: bool = True,
    ordinal_only: bool = False,
    scaler: Literal['robust', 'minmax', 'standart'] = 'robust',
    is_adult: bool = False,
) -> tuple[float, float, float]:

    # Convert the data to their desired formats

    X_train = convert_dataset(X_train)
    X_test = convert_dataset(X_test)
    y_train = convert_dataset(y_train)
    y_test = convert_dataset(y_test)    
    
    _, counts = np.unique(y_test, return_counts=True)
    minority_class_idx = counts.argmin()

    # Models
    ohe_models  = [
        KNeighborsClassifier(),
        MLPClassifier(),
        LogisticRegression(),
        SVC(probability=True),
    ]
    
 
    le_models = [
        RandomForestClassifier(),
        LGBMClassifier(verbose=-1),
    ]
    # Preprocessing for the one hot encoded 
    
    if is_adult:
        ohe_models = [KNeighborsClassifier(), LogisticRegression()]
    
    # If ordinal only make ohe model false
    ohe_model = not ordinal_only
    
    X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = _transform_data(
        X_train, X_test, y_train, y_test, discrete_columns, classification, ohe_model=ohe_model
    )
    
    F1_scores = []
    AUCROC_scores = []
    AUCPR_scores = []
    
    for model in ohe_models:
        model.fit(X_train_ohe, y_train_ohe)
        predictions = model.predict(X_test_ohe)
        predict_probas = model.predict_proba(X_test_ohe)
        if predict_probas.shape[1] == 1:
            predict_probas = np.zeros_like(predictions)
        else:
            predict_probas = predict_probas[:, 1]
        
        F1_scores.append(f1_score(y_test_ohe, predictions, pos_label=minority_class_idx, average='binary'))
        AUCROC_scores.append(roc_auc_score(y_test_ohe, predict_probas))
        AUCPR_scores.append(average_precision_score(y_test_ohe, predict_probas))
        
    # Preprocessing for the label encoded models
    X_train_le, X_test_le, y_train_le, y_test_le = _transform_data( 
        X_train, X_test, y_train, y_test, discrete_columns, classification, ohe_model=True
    )
    # Iterate over all the label encoded models and append the values calculated        
    for model in le_models:
        model.fit(X_train_le, y_train_le)
        predictions = model.predict(X_test_le)
        predict_probas = model.predict_proba(X_test_ohe)
        if predict_probas.shape[1] == 1:
            predict_probas = np.zeros_like(predictions)
        else:
            predict_probas = predict_probas[:, 1]
        
        F1_scores.append(f1_score(y_test_le, predictions, pos_label=minority_class_idx, average='binary'))
        AUCROC_scores.append(roc_auc_score(y_test_le, predict_probas))
        AUCPR_scores.append(average_precision_score(y_test_le, predict_probas))
        
    return np.mean(F1_scores), np.mean(AUCROC_scores), np.mean(AUCPR_scores)
    


def _transform_data(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.DataFrame, 
    y_test: pd.DataFrame, 
    discrete_columns: list[str],
    classification: bool = False,
    ohe_model = False,
    scaler: Literal['robust', 'minmax', 'standart'] = 'robust'  
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    continuous_columns = [col for col in X_train.columns if col not in set(discrete_columns)]


    if scaler == 'robust':
        numerical_scaler = RobustScaler()
    elif scaler == 'minmax':
        numerical_scaler = MinMaxScaler()
    elif scaler == 'standart':
        numerical_scaler = StandardScaler()
    else:
        raise ValueError(f'Scaler "{scaler}" is not an option')    
    
    numerical_pipeline = Pipeline(
        steps=[
            ('impute', SimpleImputer(strategy='mean')),
            ('preprocessing', RobustScaler()),
        ]
    )

    if ohe_model:
        categorical_pipeline = Pipeline(
            steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('preprocessing', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False)),
            ]
        )

    else:
        categorical_pipeline = Pipeline(
            steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('preprocessing', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ]
        )

    transformer = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipeline, continuous_columns),
            ('categorical', categorical_pipeline, discrete_columns),
        ],
        remainder='drop',
        sparse_threshold=0,
    )
    
    transformer.set_output(transform='pandas')

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    
    if classification:
        label_encoder = LabelEncoder()
        
        label_encoder.fit(np.concatenate((y_train, y_test)))      
           
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)

    return X_train, X_test, y_train, y_test