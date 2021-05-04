import pandas as pd
import numpy as np

PREDICTION_NAME = 'prediction'
TARGET_NAME = 'target' # this might break if they change up the format


# correlation,corr_sharpe,corr_mean,corr_std,max_drawdown,feature_exposure,max_feature_exposure
def valid4score(valid : pd.DataFrame, pred : np.ndarray, load_example: bool=True, save : bool=False) -> pd.DataFrame:
    """
    Generate new valid pandas dataframe for computing scores
    
    :INPUT:
    - valid : pd.DataFrame extracted from tournament data (data_type='validation')
    
    """
    valid_df = valid.copy() # the validation dataframe you use this to test the CORR and other values

    valid_df['prediction'] = rank_noramalize_series(pd.Series(pred)) # pred is the array of predictions your model creates for the set of validation vectors.  
    # I am unsure if this preds is a float only only between 0,1,2,3,4. 
    valid_df.rename(columns={TARGET: 'target'}, inplace=True)
    
    return valid_df

def compute_corr(df : pd.DataFrame, col1="target", col2='prediction' )->float:
    """
        Returns np.corrcoef on col1 and col2 in df
    """
    if (col1 not in df.columns) | (col2 not in df.columns):
        raise ValueException('You can only pass this function columns in valid_df')
    return np.corrcoef(df[col1], df[col2])[0, 1]

def compute_max_drawdown(validation_correlations : pd.Series):
    """
    Compute max drawdown
    
    :INPUT:
    - validation_correaltions : pd.Series
    """
    
    rolling_max = (validation_correlations + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -(rolling_max - daily_value).max()
    
    return max_drawdown

def compute_val_corr(valid_df : pd.DataFrame):
    """
    Compute rank correlation for valid periods
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    
    # all validation
    correlation = compute_corr(valid_df)
    #print("rank corr = {:.4f}".format(correlation))
    return correlation
    
def compute_val_sharpe(valid_df : pd.DataFrame):
    """
    Compute sharpe ratio for valid periods
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    # all validation
    d = valid_df.groupby('era')[['target', 'prediction']].corr().iloc[0::2,-1].reset_index()
    me = d['prediction'].mean()
    sd = d['prediction'].std()
    max_drawdown = compute_max_drawdown(d['prediction'])
    #print('sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(me / sd, me, sd, max_drawdown))
    
    return me / sd, me, sd, max_drawdown
    
def feature_exposures(valid_df : pd.DataFrame):
    """
    Compute feature exposure
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    feature_names = [f for f in valid_df.columns
                     if f.startswith("feature")]
    exposures = []
    for f in feature_names:
        fe = spearmanr(valid_df['prediction'], valid_df[f])[0]
        exposures.append(fe)
    return np.array(exposures)

def max_feature_exposure(fe : np.ndarray):
    return np.max(np.abs(fe))

def feature_exposure(fe : np.ndarray):
    return np.sqrt(np.mean(np.square(fe))) #

def compute_val_feature_exposure(valid_df : pd.DataFrame):
    """
    Compute feature exposure for valid periods
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    # all validation
    fe = feature_exposures(valid_df)
    fe1, fe2 = feature_exposure(fe), max_feature_exposure(fe)
    #print('feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(fe1, fe2))
     
    return fe1, fe2

# to neutralize a column in a df by many other columns
#         I have no idea what this method does. -P. need to read about it and write up a link to it. 

def neutralize(df, columns, by, proportion=1.0):
    scores = df.loc[:, columns]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack(
        (exposures,
         np.asarray(np.mean(scores)) * np.ones(len(exposures)).reshape(-1, 1)))

    scores = scores - proportion * exposures.dot(
        np.linalg.pinv(exposures).dot(scores))
    return scores / scores.std()


# to neutralize any series by any other series UNKNOWN
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized

def rank_noramalize_series(col:pd.Series)-> pd.Series:
    """
        Replaces unif()
        Compute the rank order of col.
        Scale each of the rankings to between 0 and 1.
        Returns a pd.Series
    """ 
    scaled_col = (col.rank(method="first") - 0.5) / len(col)
    scaled_col.index = col.index
    return scaled_col

def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)

def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME],
                                          feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: np.corrcoef(x["neutral_sub"].rank(pct=True, method="first"), x[TARGET_NAME])).mean()
    return np.mean(scores)

def compute_val_mmc(valid_df : pd.DataFrame):    
    # MMC over validation
    mmc_scores = []
    corr_scores = []
    for _, x in valid_df.groupby("era"):
        series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),
                                   pd.Series(unif(x[EXAMPLE_PRED])))
        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2)) # I have no idea what htis line does (0.29 ** 2)
        corr_scores.append(np.corrcoef(unif(x[PREDICTION_NAME]).rank(pct=True, method="first"), x[TARGET_NAME]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)

    #print("MMC Mean = {:.6f}, MMC Std = {:.6f}, CORR+MMC Sharpe = {:.4f}".format(val_mmc_mean, val_mmc_std, corr_plus_mmc_sharpe))

    # Check correlation with example predictions
    corr_with_example_preds = np.corrcoef(valid_df[EXAMPLE_PRED].rank(pct=True, method="first"),
                                          valid_df[PREDICTION_NAME].rank(pct=True, method="first"))[0, 1]
    #print("Corr with example preds: {:.4f}".format(corr_with_example_preds))
    
    return val_mmc_mean, val_mmc_std, corr_plus_mmc_sharpe, corr_with_example_preds

class ScoreCalculator:
    """
        Calcuating various metrics on the relationship between your predictions, example predictions and validation dat

        Based on this notebook: example.py of the numerai pa
    """
    def __init__(self,validation_data) -> ScoreCalculator:
        """



        """
    
        self.validation_data = ping_validation_data() 
        self.rank_normalized_validation_targets = rank_order_transfrom_columns(df=self.validation_data, col_name='target')['target'] 
        self.feature_col_names = [column_name for column_name in self.validation_data.columns if column_name.contains('feature')]
        self.example_predictions = ping_example_predictions()
        self.rank_normalized_example_predictions = rank_order_transfrom_columns(df=self.example_predictions, col_name='prediction')['prediction']
    
    
    def ping_validation_data(self):
        """
        Ping Numerai to create get the live tournament data and extact all the validation data.

        Adapted from : https://www.kaggle.com/code1110/numerai-tournament | May 3 2021
        """
        tournament_data_url = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
        tournament_df = pd.read_csv(tournament_data_url)
        valid_df = tournament_df[tournament_df["data_type"] == "validation"].reset_index(drop = True)
        feature_cols = valid_df.columns[valid_df.columns.str.startswith('feature')]

        map_floats_to_ints = {0.0 : 0, 0.25 : 1, 0.5 : 2, 0.75 : 3, 1.0 : 4}
        for col in feature_cols:
            valid_df[col] = valid_df[col].map(map_floats_to_ints).astype(np.uint8) # reduce space costs by casting features as ints
            
        valid_df["era"] = valid_df["era"].apply(lambda x: int(x[3:])) # strip the word 'era' from the era column
        valid_df.drop(columns=["data_type"], inplace=True)
        return valid_df


    def ping_example_predictions(self)-> pd.DataFrame:
        """
            Create a dataframe of Id, Prediction that are the default predictions from the example model.
            
            Used for corr with example predictions and the independence to a normal  out of the box xbgoost regressor
            id	                prediction
            n0003aa52cab36c2	0.49
            n000920ed083903f	0.49
            n0038e640522c4a6	0.53
            ...                 ...
        """
        example_predictions_url = "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_example_predictions_data.csv.xz"
        return pd.read_csv(example_predictions_url, index_col=0)

    # untested
    def richards_dependence(self, df, target_col, era_col, prediction_col) -> float: 
        """
            Measures the independendence of prediction with the targets
            
            Currently unused 
            example call:
            richards_dependence(df, 'target', 'era', 'prediction'))
            Source: Numerai Forumn user:richai @ https://forum.numer.ai/t/independence-and-sharpe/2560 | May 3 ,2021
        """  
        scores_by_era = df.groupby(era_col).apply(lambda d: d[[prediction_col, target_col]].corr()[target_col][0])
            
        # these need to be ranked within era so "error" makes sense
        df[prediction_col] = df.groupby(era_col)[prediction_col].rank(pct=True)
        df[target_col] = df.groupby(era_col)[target_col].rank(pct=True)

        df["era_score"] = df[era_col].map(scores_by_era)

        df["error"] = (df[target_col] - df[prediction_col]) ** 2
        df["1-error"] = 1 - df["error"]

        # Returns the correlation of the 1-error with the era_score
        # i.e. how dependent/correlated each prediction is with its era_score
        return df[["1-error", "era_score"]].corr()["era_score"][0]


    def rank_noramalize_series(self, col:pd.Series)-> pd.Series:
        """
            Compute the rank ordering of col. Scale each element of col between 0 and 1 based on their relative size
            Returns: a pd.Series
        """ 
        scaled_col = (col.rank(method="first") - 0.5) / len(col)
        scaled_col.index = col.index
        return scaled_col


    def rank_order_transfrom_columns(self, df: pd.DataFrame, col_name: str)-> pd.DataFrame:
        """
            Returns a copy of df with df[col_name], rank normalized
        """
        df_copy = df.copy()
        df_copy['prediction'] = rank_noramalize_series(df_copy['prediction'])
        return df_df_copy


 


# you pass this Object a model_prediction_df
# that has 