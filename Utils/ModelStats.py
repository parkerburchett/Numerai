import pandas as pd
import numpy as np

class ModelStats():
  """
  An object that tracks Hyper Parameters, Time Costs and Scores. 
  You must globally define ROUND_NUMBER and PATH_TO_SAVE_SCORES.
  Note you might want to write a ModelStatsFactory(ROUND_NUMBER,PATH_TO_SAVE_SCORES) to locally define the variables. It is not tested
  """
  def __init__(self, model, scores:dict, total_time):
        self.model = model 
        self.hyperparams = model.get_params() 
        self.scores = scores 
        self.total_time = total_time
        self.params_scores_df = None 


  def create_params_scores_df(self) -> None:
    """
    Create a DataFrame Representing the Hyper Parameters and Scores of this model.
    """
    if self.params_scores_df == None:
      all_stats_dict = {}
      all_stats_dict['total_time'] = self.total_time
      all_stats_dict['round_number'] = ROUND_NUMBER
      all_stats_dict.update(self.hyperparams) # dict.update(dict) merges two dictionaries
      all_stats_dict.update(self.scores)
      DECIMALS = 4 
      for key in all_stats_dict.keys():
          try:
            all_stats_dict[key] = [round(all_stats_dict[key], DECIMALS)]
          except:
            all_stats_dict[key] = [all_stats_dict[key]] # overwrite them as a 1 element list so that you can convert them into a DataFrame
      self.params_scores_df = pd.DataFrame.from_dict(all_stats_dict)

  
  def save_hyperparams_scores_to_google_drive_tabular(self)-> None:
    """
        Save to the scores and hyper parameter to Google Drive based on PATH_TO_SAVE_SCORES
    """
    self.create_params_scores_df()
    disk_df = pd.read_csv(PATH_TO_SAVE_SCORES)
    old_scores_and_new_scores_df = merge_dfs_horizontally(disk_df, self.params_scores_df)
    old_scores_and_new_scores_df.to_csv(PATH_TO_SAVE_SCORES, index=False)

    try:
      with open(PATH_TO_SAVE_SCORES, 'r') as scores_file:
          lines = scores_file.readlines()
          if len(lines) == 0:
            print("the file does not exist. You are good to save your first score df")
    except:
      self.params_scores_df.to_csv(PATH_TO_SAVE_SCORES, index=False)
           

def merge_dfs_horizontally(df1 : pd.DataFrame, df2: pd.DataFrame)-> pd.DataFrame:
  merged_df = pd.concat([df1, df2], axis=0)
  return merged_df