import pandas as pd
import numpy as np

def rank_noramalize_series(col:pd.Series)-> pd.Series:
    """
        Compute the rank order of col.
        Scale each of the rankings to between 0 and 1.
        Returns a pd.Series
    """ 
    scaled_col = (col.rank(method="first") - 0.5) / len(col)
    scaled_col.index = col.index
    return scaled_col


d = {'col1': [1, 2,5], 'col2': [5, 4,2]}
df = pd.DataFrame(data=d)
print(df['col1'])
print(rank_noramalize_series(df['col1']))