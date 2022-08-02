import datetime
import time
import os
from typing import Callable

from numerapi import NumerAPI
import pandas as pd
import lightgbm as lgb
import optuna
import numereval
import numpy as np
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_param_importances



TRAIN_FILE_PATH = 'train_int8.parquet'
VALID_FILE_PATH = 'validation_int8.parquet'
TARGET = 'target'
PREDICTION = 'prediction'

# SAVE_DIR = "/content/drive/MyDrive/CryptoCurency/Numerai/Notebooks/Explore New Data/post 7 trees/v4_feature_subset_tests"

# FIRST_N_FOLDS = 3
# NUM_OPTUNA_TRIALS = 200
# NUM_FEATURES = 200
# N_ESTIMATORS = 500

def _build_save_path(save_dir: str):
  unix_time_now = int(time.mktime(datetime.datetime.now().timetuple()))
  return f'{save_dir}/v_4_feature_subset_{unix_time_now}.csv'


def load_hp_search_data() -> None:
  """Loads the data from numerai into disk to use later in hp searching"""
  NumerAPI().download_dataset(filename='v4/train_int8.parquet', dest_path=TRAIN_FILE_PATH, round_num=326)
  NumerAPI().download_dataset(filename='v4/validation_int8.parquet', dest_path=VALID_FILE_PATH, round_num=326)


def build_validation_dfs(valid_df: pd.DataFrame) -> list:
  """Return a list of eras to use in each validation set"""
  eras = valid_df['era'].unique()[4:] # avoids overlap due to these weeks overlapping with the traing data
  valid_0 = eras[:86] 
  valid_1 = eras[86:86*2]
  valid_2 = eras[86*2:86*3]
  valid_3 = eras[86*3:86*4]
  valid_4 = eras[86*4:]
  validation_era_groups = [valid_0, valid_1, valid_2, valid_3, valid_4]
  
  valid_dfs = []
  for validation_era_group in validation_era_groups:
    valid_dfs.append(valid_df[valid_df['era'].isin(list(validation_era_group))].copy())
  return valid_dfs


def evaluate_model(model: lgb.LGBMRegressor, feature_sub_list: list, valid_dfs: list,
                   model_eval_func: Callable): #model_eval_func takes [pd.DataFrame, float]
    """Returns the model fittness, df of summary of results"""
    cv_results = {}
    for fold_num, sub_valid_df in enumerate(valid_dfs):
        sub_valid_df[PREDICTION] = model.predict(X=sub_valid_df[feature_sub_list])
        cv_results[f'cv_ {fold_num}'] = numereval.evaluate(sub_valid_df)['metrics']

    model_cv_summary = pd.DataFrame(cv_results)
    model_fitness = model_eval_func(model_cv_summary)
    return model_fitness, model_cv_summary 


def save_model_performace(model_params: dict, feature_sub_list:list, model_summary: pd.DataFrame,
                          records: list[dict], save_path: str) -> None:
    """Save model params, feature_sub_list to disk. """
    cleaned_model_data = model_summary.mean(axis=1).round(6).to_dict() #
    cleaned_model_data['feature_sub_list'] = str(feature_sub_list)
    cleaned_model_data.update(model_params)
    records.append(cleaned_model_data)
    pd.DataFrame.from_records(records).to_csv(save_path)


def _build_model_params(trial, n_estimators: int, max_learning_rate: float, max_num_leaves: float,
                        max_col_sample_bytree: float, max_feature_fraction: float, 
                        max_bagging_fraction: float, max_bagging_freq: int) -> dict:
    """
    Build a set of hyper params for a lgbm model based on the optuna trial object
    """
    return {
        "objective": "regression", "metric": "mse", "verbosity": -1, "boosting_type": "gbdt",
        "n_jobs": -1, 'n_estimators': n_estimators,
        'learning_rate': trial.suggest_float("learning_rate", .0001, max_learning_rate, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, max_num_leaves),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0, max_col_sample_bytree),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, max_feature_fraction),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, max_bagging_fraction),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, max_bagging_freq)
        }


def run_optuna_on_feature_subset(train_df: pd.DataFrame, valid_dfs: list[pd.DataFrame], save_dir: str, 
                                 features: list[str], target: str,  num_features: int, n_estimators: int,
                                 num_optuna_trials: int, max_learning_rate: float,  max_num_leaves: float, 
                                 max_col_sample_bytree: float, max_feature_fraction: float, max_bagging_fraction: float,
                                 max_bagging_freq: int, model_eval_func: Callable) -> None:
    """
    Runs optuna on these hyper params and save the results to disk as a .csv
    """
    records = []
    save_path = _build_save_path(save_dir)
    feature_sub_list = np.random.choice(features, num_features, replace=False)
    def objective(trial):
        model_params = _build_model_params(trial, n_estimators, max_learning_rate, max_num_leaves,
                                           max_col_sample_bytree, max_feature_fraction, max_bagging_fraction,
                                           max_bagging_freq)
        model = lgb.LGBMRegressor(**model_params).fit(X=train_df[feature_sub_list], y=train_df[target])
        model_fitness, model_summary = evaluate_model(model, feature_sub_list, valid_dfs, model_eval_func)
        save_model_performace(model_params, feature_sub_list, model_summary, records, save_path)
        print(model_summary)
        return model_fitness
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_optuna_trials)
    print(f"Number of finished trials: {len(study.trials)} Best trial: {study.best_trial}")
    plot_param_importances(study)
    plt.show()
