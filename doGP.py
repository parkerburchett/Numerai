### main
"""
	Notes on first run.
	
	When you run this on your computer it makes everything else you do very slow. 
	
	How do you restrict the amount of memory that this program can use. 
	
	Maybe run it on a virual machine?
	
	https://www.geeksforgeeks.org/python-how-to-put-limits-on-memory-and-cpu-usage/

"""




import numerapi
import numpy as np
import pandas as pd
import json
import pickle
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer

def create_function_set():
  """
    The individual Atomic Functions to be used as the parts of the linear transformations
  """
  tanh = make_function(np.tanh,'tanh', arity=1)
  divide_by_two = make_function(lambda col: np.divide(col,2),'divide_by_two', arity=1)
  #average = make_function(lambda a,b: np.average(a,b) , 'average', arity=2) # broken. Now unsure why.
  function_set = ['add', 'sub', 'mul', 'div','neg', tanh, divide_by_two] 
  return function_set

def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]

def score(df): # copied from example.py from Numerai
    return correlation(df['prediction'], df['target'])

def atomic_program_evaluation(atomic_program, valid_X, valid_y):
  """
      Pass this function a program and it returns a dict of the evaluation of the program.
      'program_obj': the atomic program object
      'program_str': String repersentation of the program
      'corr': the person validation corr on the unseen validation data for this round.
  """
  res = pd.DataFrame()
  res['prediction'] = atomic_program.execute(valid_X.values)
  res['target'] = valid_y.values  # defined globally
  outcome = dict()
  outcome['program_obj'] = atomic_program
  outcome['program_str'] = str(atomic_program)
  outcome['corr'] = score(res)
  return outcome

def create_new_unfit_symbolic_transformer(function_set:list, feature_cols, verbose=True, simple=False):

  if not simple:
    new_transformer = SymbolicTransformer(verbose=verbose, # these were choose with no real reason in mind.
                            feature_names=feature_cols,
                            generations=20,
                            metric='spearman', 
                            parsimony_coefficient =.0005, 
                            population_size= 1000, 
                            function_set=function_set,
                            n_jobs=-1,
                            max_samples=.5, # only test fitness on a 50% of the data.
                            init_depth = (5,7),
                            low_memory=True,
                            stopping_criteria = .035,
                            tournament_size=100)
  else:
      new_transformer = SymbolicTransformer(verbose=verbose,
                            feature_names=feature_cols,
                            generations=3,
                            metric='spearman',
                            low_memory=True,
                            parsimony_coefficient =.0005, 
                            population_size= 500, 
                            function_set=function_set,
                            n_jobs=-1,
                            init_depth = (2,3),
                            stopping_criteria = .035,
                            tournament_size=50)

  return new_transformer #unfit_transformer = create_new_unfit_symbolic_transformer()

def setup(): # you cannot pickle the setup.
  napi = numerapi.NumerAPI()
  current_round = napi.get_current_round()
  napi.download_current_dataset(unzip=True)
  train_df = pd.read_csv(f'numerai_dataset_{current_round}/numerai_training_data.csv', index_col=0)
  tournament_df = pd.read_csv(f'numerai_dataset_{current_round}/numerai_tournament_data.csv', index_col=0)
  print('you have read the data into memory')
  feature_cols = [c for c in train_df.columns if c.startswith("feature")]
  X = train_df[feature_cols]
  y = train_df['target']
  valid_df = tournament_df[tournament_df["data_type"] == "validation"].reset_index(drop = True)
  valid_X = valid_df[feature_cols]
  valid_y = valid_df['target']
  function_set = create_function_set()
  return train_df, tournament_df, valid_df, feature_cols, X, y, valid_X, valid_y, function_set


def main():
  print('in setup')
  train_df, tournament_df, valid_df, feature_cols, X, y, valid_X, valid_y, function_set = setup() # takes 2 minutes on colab on high ram

  print("done with setup")
  def evolve_new_atomic_programs(valid_X, valid_y):
    """
      Defined internally to not have to repass params.I am not sure if this is faster
    """
    transformer = create_new_unfit_symbolic_transformer(function_set=function_set, feature_cols=feature_cols, simple=False)
    print('now EVOLVING')
    transformer.fit(X,y)
    return [atomic_program_evaluation(prog, valid_X, valid_y) for prog in transformer] # returns a list of dicts.

  for batch in range(3):
    print(f'BATCH {batch}')
    log_pickle_location = "log_0.pkl"  

    #Read already evolved atomic programs from disk.
    try:
      atomic_programs_file = open(log_pickle_location,'rb')
      list_of_atomic_programs = pickle.load(atomic_programs_file)
      atomic_programs_file.close()
      if type(list_of_atomic_programs) is not list:
        raise ValueError() # just to get to except. Bad code practice
    except:
      list_of_atomic_programs = []

    # add those atomic programs to your system.
    new_atomic_programs = evolve_new_atomic_programs(valid_X, valid_y)
    print('you successfully evolved new atomic programs')
    list_of_atomic_programs.extend(new_atomic_programs)

    print('look at the models you evolved')
    for s in new_atomic_programs:
      print(s)

    # write the new programs to disk.
    atomic_programs_file = open(log_pickle_location,'wb') #
    pickle.dump(list_of_atomic_programs, atomic_programs_file)
    atomic_programs_file.close()
    print('You now have this have this many atomic programs')
    print(len(list_of_atomic_programs))
  

    

main() # next call there should be 20
