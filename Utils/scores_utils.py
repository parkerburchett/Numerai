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


# this is the main method. The rest are just called interanlly. 
def score_summary(valid_df : pd.DataFrame):
    score_dict = {}

    try:
        score_dict['correlation'] = compute_val_corr(valid_df)
    except:
        print('ERR: computing correlation')
    try:
        score_dict['corr_sharpe'], score_dict['corr_mean'], score_dict['corr_std'], score_dict['max_drawdown'] = compute_val_sharpe(valid_df)
    except:
        print('ERR: computing sharpe')
    try:
        score_dict['feature_exposure'], score_dict['max_feature_exposure'] = compute_val_feature_exposure(valid_df)
    except:
        print('ERR: computing feature exposure')
    # try:
    #     score_dict['mmc_mean'], score_dict['mmc_std'], score_dict['corr_mmc_sharpe'], score_dict['corr_with_example_xgb'] = compute_val_mmc(valid_df)
    # except:
    #     print('ERR: computing MMC')
    
    return score_dict

class ScoreCalculator:
    """
        This is an object that holds the methods to compute various scores for evaluating how a model performs on unseen Test Data

        You need to pass it the current round VALIDATION_DATA at construction

    """
    def __init__(self, VALIDATION_DATA):
        self.VALIDATION_DATA = VALIDATION_DATA # a DataFrame of id, era, features1 ....featurnN, Target
        self.FEATURES = [column_name for column_name in VALIDATION_DATA.columns if column_name.contains('feature')]
        self.EXAMPLE_PRED = None # this is the default preditions for this round # you should ping the numerai
    
    def ping_example_predictions(self):
        """
            Get the example predictions for the current round from Numerai.
            stub
        """
        return 0

    def create_default_score_dict(self, predictions: pd.Series)-> dict:
        """
            Return a dictionary object that corrosponds to the "diagnostics" tab on numerai 
        """

        score_dict = {}

        try:
            score_dict['correlation'] = compute_val_corr(valid_df)
        except:
            print('ERR: computing correlation')
        try:
            score_dict['corr_sharpe'], score_dict['corr_mean'], score_dict['corr_std'], score_dict['max_drawdown'] = compute_val_sharpe(valid_df)
        except:
            print('ERR: computing sharpe')
        try:
            score_dict['feature_exposure'], score_dict['max_feature_exposure'] = compute_val_feature_exposure(valid_df)
        except:
            print('ERR: computing feature exposure')
        # try:
        #     score_dict['mmc_mean'], score_dict['mmc_std'], score_dict['corr_mmc_sharpe'], score_dict['corr_with_example_xgb'] = compute_val_mmc(valid_df)
        # except:
        #     print('ERR: computing MMC')

        return score_dict