import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from ..preprocessing.data_transformer import DataTransformer

# Minimize
# NOTE: Look at the twang package in R (It uses xgboost to calculate propensity scores)
class PropensityScore:
    @classmethod
    def compute(
        cls,     
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        discrete_columns: list[str] = [],
        continuous_columns: list[str] | None  = None
    ) -> float:

        # Transform the data


        n_real = len(real_data)
        n_synhtetic = len(synthetic_data)
        
        # Generate the training set and its labels
        X = pd.concat([real_data, synthetic_data], ignore_index=True)
        y = np.concatenate([np.ones(shape=(n_real,)), np.zeros(shape=(n_synhtetic,))])
        
        propensity_scores = []
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for train_idx, test_idx in kf.split(X, y):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            transformer = DataTransformer(discrete_columns=discrete_columns)
            
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)
            
            # Create the classifier model
            classifier = LogisticRegression(solver='liblinear')
            # Fit the model and extract probabilites
            classifier.fit(X_train, y_train)
            prob_predictions = classifier.predict_proba(X_test)[:, 1]
            # Propensity_Score(X) = 1/n sum_n (pi - 0.5)^2
            propensity_score = mean_squared_error(prob_predictions, np.full(shape=(len(prob_predictions),), fill_value=0.5))
            propensity_scores.append(propensity_score)

        return np.mean(propensity_scores)

