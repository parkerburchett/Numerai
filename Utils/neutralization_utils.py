from numereval import numereval
import numerapi
import pandas as pd
import numpy as np

def neutralize_predictions_by_features(predictions:pd.Series, tournament_df:pd.DataFrame, proportion=1):
  """
    Take the predictions and with numereval.neutralize_series() neutralize them by the proportion
    returns the rank normalized Series of prediction
  """
  normalized_predictions = predictions
  features = [f for f in tournament_df.columns if 'feature' in f]
  for f in features:
    normalized_predictions = numereval.neutralize_series(normalized_predictions, tournament_df[f], proportion=proportion)
  return normalized_predictions.rank(pct=True, method="first")

def submit_neutralized_predictions(predictions:pd.Series,
                                   tournament_df:pd.DataFrame, 
                                   proportion:float,
                                   model_id:str,
                                   napi: numerapi.numerapi.NumerAPI,
                                   creds:dict,
                                   predictions_file_name=None):
  """
      Neutralize the `predictions` by all the features in `tournament_df` by `proportion` 
      Then submit the rank normalized predictions to `model_id` with `napi` based on the model_id value in creds
      Only takes 2 minutes on the old data
  """
  current_round = napi.get_current_round()

  if proportion > 0:
    neutralized_predictions = neutralize_predictions_by_features(predictions=predictions,
                                                                tournament_df=tournament_df,
                                                                proportion=proportion)
  else:
    neutralized_predictions = predictions
    
  neutralized_pred_df = pd.DataFrame(index=neutralized_predictions.index,
                                     data=neutralized_predictions.values,
                                     columns= [['prediction']])
  
  if predictions_file_name is None:
    predictions_file_name = f'/content/round_{current_round}_model_{model_id}_predictions.csv'
  
  neutralized_pred_df.to_csv(predictions_file_name, index=True, header=['prediction'])
  napi.upload_predictions(predictions_file_name, model_id=creds[model_id], version=1)
  print(f'Successfully submitted {model_id}')
