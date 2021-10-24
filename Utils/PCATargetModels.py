import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import lightgbm as lgb

import warnings # should remove the future warning error. Untested
warnings.simplefilter(action='ignore', category=FutureWarning)

class PCATargetModelWapper:
    """
    A wapper for a sklearn-esque model that is trained to predict the PCA of a subset of the new target cols.

    Example:
    pca_target_model = PCATargetModelWapper(model = lgb.LGBMRegressor(), feature_cols=features[:210], target_cols=targets_20)
    pca_target_model.fit(training_data, training_data[targets])
    y_preds = pca_target_model.predict(validation_data)
    """
    def __init__(self, model, feature_cols, target_cols):
        """
        params: 
        model: a model object that must implement .fit() and .predict(). Only tested with lgb.LGBMRegressor()
        feature_cols: a list of feature columns used to train self.model 
        target_cols: a list of target columns used to train self.pca_model
        """
        self.model = model
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.pca_model = None


    def fit(self, X:pd.DataFrame, targets_df:pd.DataFrame) -> None:
        """
        Fit the pca_model and model on X and targets_df.

        example:
        pca_target_model.fit(training_data, training_data[targets])

        params:
        X: The data to train the model on. Almost always X=training_data
        targets_df: the dataframe of targets 
        """
        if X.shape[0] != targets_df.shape[0]:
            raise ValueError(f'X must have the same number of rows as targets_df.\nYou passed: X.shape {X.shape} target_dfs.shape {target_dfs.shape}')
        
        features_are_valid = np.all([feature in X.columns for feature in self.feature_cols])
        if not features_are_valid:
            raise ValueError("Some features in self.feature_cols are not in X.columns")

        targets_are_valid = np.all([target in targets_df.columns for target in self.target_cols])
        if not targets_are_valid:
            raise ValueError("Some targets in self.target_cols are not in target_dfs.columns")
        
        filled_targets_df = targets_df.loc[:, self.target_cols].fillna(0.5)
        self.pca_model = PCA(1).fit(filled_targets_df)

        pca_transformed_targets = self.pca_model.transform(filled_targets_df).reshape(filled_targets_df.shape[0])
        self.model.fit(X[self.feature_cols], pca_transformed_targets)


    def _inverse_transform_predictions(self, pca_transformed_predictions:np.array):
        """
            Compute the inverse transformation of self.pca_model on pca_transformed_predictions

            pca_transformed_predictions: an array of the pre inverse predictions of self.model

            Returns the 0th column of the inverse transformed predictions. This the 'target column'
            Note: this breaks if they change the order of the targets
        """
        inverse_transformed_targets = [self.pca_model.inverse_transform(p)[0] for p in pca_transformed_predictions]
        return np.array(inverse_transformed_targets)[:,0]


    def predict(self, X:pd.DataFrame):
        """
            Predict the the PCA transformed target of X and then do the inverse transformation
        """
        features_are_valid = np.all([feature in X.columns for feature in self.feature_cols])
        if not features_are_valid:
            raise ValueError("Some features in self.feature_cols are not in X.columns")

        pca_transformed_predictions = self.model.predict(X[self.feature_cols])
        return self._inverse_transform_predictions(pca_transformed_predictions)


    def get_params(self):
        """
            Returns a dictionary of this model's params
        """

        regressor_params = self.model.get_params()


        wrapper_params = {
            'feature_cols':self.feature_cols,
            'target_cols':self.target_cols
        }

        params = {
            'regressor_params': regressor_params,
            'wrapper_params': wrapper_params
        }
        return params

    def __str__(self):
        return str(self.model) + str(self.model.get_params()) + f'\n feature_cols: {self.feature_cols[:3]}...' + f'\n target_cols: {self.target_cols[:3]}...'  





class PCATargetEnsemble:
    """
        A weighted Ensemble of PCATargetModelWapper objects.
    """

    def __init__(self, PCATargetModelWappers:list, weights:np.array):
        if len(PCATargetModelWappers) != weights.shape[0]:
            raise ValueError(f'The number of models and the number of weights must be equal\n Num models: {len(PCATargetModelWappers)} Num weights: {weights.shape[0]}')
        if weights.sum() != 1:
            raise ValueError(f'weights must sum to 1 currently sums to {weights.sum()}')

        self.models = PCATargetModelWappers
        self.weights = weights

    def fit(self, X, targets_df):
        [model.fit(X,targets_df) for model in self.models]

    def predict(self, X):
        prediction_matrix = np.array([model.predict(X) for model in self.models])
        weighted_predictions = prediction_matrix.T.dot(self.weights)
        return weighted_predictions



def testing():
    feature_cols = ['feature3', 'feature_33']*100

    model = PCATargetModelWapper(model=lgb.LGBMRegressor(n_estimators=1200),

                                feature_cols=feature_cols,
                                target_cols=['target_1', 'target_23'])
    print(model)

# testing()

