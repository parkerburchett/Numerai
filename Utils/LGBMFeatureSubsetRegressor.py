"""
Wrapper class around lightgbm.LGBMRegressor that lets you set target and features at model init and then pass the
entire train_df or valid_df to the predict or fit method

example usage
model = LGBMFeatureSubsetRegressor(features, target)
model.fit(train_df)
model_path = 'local_model.json'
model.save_model(model_path)
print(model.predict(valid_dfs[0])[:5])
new_model = LGBMFeatureSubsetRegressor.load_model(model_path)
new_model.predict(valid_dfs[0])[:5]

"""

import json

import lightgbm
import pandas as pd


class LGBMFeatureSubsetRegressor(lightgbm.LGBMRegressor):
  """
  Simple wrapper around a feature the LGBMRegressor model to add an extra param: features that will limit the features
  used to train the model
  see https://github.com/microsoft/LightGBM/issues/5010
  """
  def __init__(
          self, 
          features,
          target, 
          boosting_type= 'gbdt',
          num_leaves=31,
          max_depth= -1,
          learning_rate=0.1,
          n_estimators=100,
          subsample_for_bin=200000,
          objective=None,
          class_weight=None,
          min_split_gain=0.,
          min_child_weight=1e-3,
          min_child_samples=20,
          subsample=1.,
          subsample_freq=0,
          colsample_bytree=1.,
          reg_alpha=0.,
          reg_lambda=0.,
          random_state=None,
          n_jobs=-1,
          importance_type='split',
          **kwargs
      ):
      self.features = sorted(features)
      self.target = target
      super().__init__(
          boosting_type=boosting_type,
          num_leaves=num_leaves,
          max_depth=max_depth,
          learning_rate=learning_rate,
          n_estimators=n_estimators,
          subsample_for_bin=subsample_for_bin,
          objective=objective,
          class_weight=class_weight,
          min_split_gain=min_split_gain,
          min_child_weight=min_child_weight,
          min_child_samples=min_child_samples,
          subsample=subsample,
          subsample_freq=subsample_freq,
          colsample_bytree=colsample_bytree,
          reg_alpha=reg_alpha,
          reg_lambda=reg_lambda,
          random_state=random_state,
          n_jobs=n_jobs,
          importance_type=importance_type,
          **kwargs
        )
      
  def predict(self, df: pd.DataFrame ):
      return super().predict(X=df[self.features])

  def fit(
    self,
    train_df: pd.DataFrame,
    sample_weight=None,
    init_score=None,
    eval_set=None,
    eval_names=None,
    eval_sample_weight=None,
    eval_init_score=None,
    eval_metric=None,
    feature_name='auto',
    categorical_feature='auto',
    callbacks=None,
    ):
    super().fit(
        X=train_df[self.features],
        y=train_df[self.target],
        sample_weight=sample_weight,
        init_score=init_score,
        eval_set=eval_set,
        eval_names=eval_names,
        eval_sample_weight=eval_sample_weight,
        eval_init_score=eval_init_score,
        eval_metric=eval_metric,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
        callbacks=callbacks,
    )
    return self

  def save_model(self, model_path: str) -> None:
    """
    Saves this model to model_path as a json
    
    model_params dict: is the hyper params used at __init__()
    model_str str: is the string repersentation of the model
    
    """
    with open(model_path, 'w') as f:
      if model_path[-4:] != 'json':
        raise ValueError('can only save model to a json')
      json.dump({'model_str': self.booster_.model_to_string(), 'model_params' : self.get_params()}, f)

  @classmethod
  def load_model(cls, model_path: str) -> None:
    if model_path[-4:] != 'json':
      raise ValueError('can only load model from a json')
    with open(model_path, 'r') as f:
      model_data = json.load(f)
      model = cls(**model_data['model_params'])
      model._Booster = lightgbm.Booster({'model_str': model_data['model_str']})
      model._n_features = len(model.features) # this trick the model into think it has been fit already
      return model
