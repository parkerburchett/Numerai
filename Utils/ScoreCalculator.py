import pandas as pd
import numpy as np


class ScoreCalculator:
    """
        Calcuating various metrics on the relationship between your predictions, example predictions and validation data.



        Primarily based on: example.py 
    """
    def __init__(self,validation_data) -> ScoreCalculator:
        """



        """
        self.validation_data = ping_validation_data() 
        self._rank_normalized_validation_targets = rank_order_transfrom_columns(df=self.validation_data, col_name='target')['target'] 
        self._feature_col_names = [column_name for column_name in self.validation_data.columns if column_name.contains('feature')]
        self.example_predictions = ping_example_predictions()
        self._rank_normalized_example_predictions = rank_order_transfrom_columns(df=self.example_predictions, col_name='prediction')['prediction']
    
    # called during init
    def ping_validation_data(self) -> pd.DataFrame:
        """
        Ping Numerai to create get the live tournament data and extact all the validation data.

        Copied from : https://www.kaggle.com/code1110/numerai-tournament | May 3, 2021
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

    # called during init # broken You need to specify that this is example preds over the entire df
    def ping_example_predictions(self)-> pd.DataFrame:
        """
            Create a dataframe of Id, Prediction that are the default predictions from the example model.
            
            Used for corr with example predictions and the independence to a normal  out of the box xbgoost regressor
            id	                prediction
            n0003aa52cab36c2	0.49
            n000920ed083903f	0.49
            n0038e640522c4a6	0.53
            ...                 ...

            # 
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

    # unused
    def rank_order_transfrom_columns(self, df: pd.DataFrame, col_name: str)-> pd.DataFrame:
        """
            Returns a copy of df with df[col_name], rank normalized between [0,1]
        """
        df_copy = df.copy()
        df_copy['prediction'] = rank_noramalize_series(df_copy['prediction'])
        return df_df_copy


    def compute_validation_corr(self, pred: pd.Series)-> float:
        """
            pred: your predictions on the validation data.
            Compute your corr on the validation data.
        """
        rank_normalized_preds = rank_noramalize_series(pred)
        return _compute_corr(self._rank_normalized_validation_targets, rank_normalized_preds)


    def _compute_corr(self, a: pd.Series, b: pd.Series )->float:
        """
            Returns np.corrcoef on a and b. pass this only ranked correlations
        """
        return np.corrcoef(a, b)[0, 1]


    def compute_validation_std(self, pred: pd.Series) -> float:
        """
            Returns the Standard Deviation of corr by era.
            can be made to be more efficient
        """
        pred_valid_df = self.merge_pred_valid_df(pred) 
        pred_valid_df_corr = pred_valid_df.groupby("era").apply(score)
        return pred_valid_df_corr.std()


    def compute_validation_sharpe(self, pred:pd.Series ):
        """
            Computes your sharpe corr socre on each era 
            sharpe = average corr by era / std dev of corr by era 
        """
        pred_valid_df = self.merge_pred_valid_df(pred) 
        pred_valid_df_corr = pred_valid_df.groupby("era").apply(score)
        pred_valid_df_corr_mean = pred_valid_df_corr.mean()
        pred_valid_df_corr_std =compute_validation_std(pred)
        return pred_valid_df_corr_mean / pred_valid_df_corr_std
        
    
    def score(self, df: pd.DataFrame)-> float:
        """
            utility to compute corr on a grouping of self._validation_data 
        """
        return _compute_corr(df['target'], df['prediction'])


    def compute_max_drawdown(self, pred: pd.Series)-> float:
        """
            Copied from Numerai's example_model.py
            Max drawdown is the "largest cumulative between any two eras in terms of validation correlation"
            Is an estimate of risk
        """
        validation_correlations = self.compute_validation_corr(pred).groupby("era").apply(score) # this needs to be stored in the class variables do avoid doing it twice
        rolling_max = (validation_correlations + 1).cumprod().rolling(window=100, min_periods=1).max()
                                                                    
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -(rolling_max - daily_value).max()
        return max_drawdown


    def compute_feature_exposure(self, pred:pd.Series)-> float:
        """
            The maximum corrilatiosn your predictions have with any single feature
            Copied from Numerai's example_model.py
        """
        pred_valid_df = self.validation_data # unclear if the default to 
        pred_valid_df['prediction'] = rank_noramalize_series(pred) # add prediction column

        feature_names = [f for f in self.validation_data.columns if f.startswith("feature")]
        feature_exposures = validation_data[feature_names].apply(lambda d: correlation(pred_valid_df['prediction'], d), axis=0)
                                                             
        max_feature_exposure = np.max(np.abs(feature_exposures))
        return max_feature_exposure


    def compute_feature_neutral_mean(self, pred:pd.Series)-> float:
        """
            The mean of your per era correlation after your predictions have been neutralized to all the features
            Copied from Numerai's example_model.py
        """
        pred_valid_df = self.validation_data # unclear if the default to 
        pred_valid_df['prediction'] = rank_noramalize_series(pred) # add prediction column

        feature_cols = [c for c in df.columns if c.startswith("feature")]
        pred_valid_df.loc[:, "neutral_sub"] = neutralize(pred_valid_df, ['prediction'],
                                            feature_cols)['prediction']
        
        # I made aded the rank normalize within the lambda
        scores = df.groupby("era").apply(
            lambda x: self._compute_corr(rank_noramalize_series(x["neutral_sub"]), rank_noramalize_series(x['target']))).mean()
        return np.mean(scores)

    #untested
    def neutralize(self, df, columns, by, proportion=1.0):
        """
            Copied as is from example_model.py
        """
        scores = df.loc[:, columns]
        exposures = df[by].values

        # constant column to make sure the series is completely neutral to exposures
        exposures = np.hstack(
            (exposures,
            np.asarray(np.mean(scores)) * np.ones(len(exposures)).reshape(-1, 1)))

        scores = scores - proportion * exposures.dot(
            np.linalg.pinv(exposures).dot(scores))
        return scores / scores.std()
    
    #untested
    def compute_mmc_stats(self, pred:pd.Series) -> Tuple:
        """
            Using example predictions as an estimate for the meta model, compute mmc stats
            Copied from example_model.py
            returns val_mmc_mean, corr_plus_mmc_sharpe, corr_plus_mmc_sharpe_diff 

            Not refractored. Copied as is. Only variable and function names are changed

        """
        pred_valid_df = self.validation_data # unclear if the default to 
        pred_valid_df['prediction'] = rank_noramalize_series(pred) # add prediction column
        pred_valid_df['ExamplePreds'] = self.example_predictions
        mmc_scores = []
        corr_scores = []

        for _, x in validation_data.groupby("era"):
            series = self.neutralize_series(pd.Series(self.rank_noramalize_series(x['prediction'])),
                                    pd.Series(self.rank_noramalize_series(x["ExamplePreds"])))
            mmc_scores.append(np.cov(series, x['target'])[0, 1] / (0.29 ** 2))
            corr_scores.append(correlation(self.rank_noramalize_series(x['prediction']), x['target']))

        val_mmc_mean = np.mean(mmc_scores)
        val_mmc_std = np.std(mmc_scores)
        val_mmc_sharpe = val_mmc_mean / val_mmc_std
        corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
        corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
        corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
        corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

        print(
            f"MMC Mean: {val_mmc_mean}\n"
            f"Corr Plus MMC Sharpe:{corr_plus_mmc_sharpe}\n"
            f"Corr Plus MMC Diff:{corr_plus_mmc_sharpe_diff}"
        )
        return  val_mmc_mean, corr_plus_mmc_sharpe, corr_plus_mmc_sharpe_diff

    #untested
    def neutralize_series(self, series, by, proportion=1.0):
        """
            Copied from example_model.py
            not refactored
        """
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


    def compute_corr_with_example_preds(self, pred:pd.Series) -> float:
        """
            Returns the rank corrilation of your predictions (pred) with the example predictions
        """
        ranked_example_preds = self.rank_noramalize_series(self.example_predictions)
        ranked_user_preds = self.rank_noramalize_series(pred)
        return self._compute_corr(ranked_example_preds, ranked_user_preds)
    

    def merge_pred_valid_df(self,pred: pd.Series):
        """
            Add your predictions to self.validation_data in order to make calcuating the answers more efficnet 
        """

        pred_valid_df = self.validation_data # unclear if the default to 
        pred_valid_df['prediction'] = rank_noramalize_series(pred)

        return pred_valid_df        


    def compute_per_era_corr(self, pred: pd.Series) -> list:
        """
            Returns a list of tuples for representing (era, corr for era)
        """
        pred_valid_df = self.merge_pred_valid_df(pred)
        era_corr_list = []
        eras = list(pred_valid_df['era'].unique())
        for era in eras:
            local_era_targets = np.array(valid_df[valid_df['era'] == era]['target'])
            local_era_predictions = np.array(valid_df[valid_df['era'] == era]['prediction'])
            era_corr = self._compute_corr(local_era_targets, local_era_predictions)
            era_corr_list.append((era,era_corr))
        
        return era_corr_list
