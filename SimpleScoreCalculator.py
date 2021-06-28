import pandas as pd
import numpy as np


class ScoreCalculator:
    """
        Calcuating various metrics on the relationship between your predictions, example predictions and validation data.
        Call compute_numerai_diagnostics(a pd.Series of your predictions on the validation data)
        Add method to compute more differnent diagonistcs
        Currently not throughly tested.
        Primarily based on: example.py 
    """
    def __init__(self, validation_data, example_preds):
        """
          validation_data['rank_target'] is handeled in ping validation data
        """

        self.validation_data = validation_data
        self._rank_normalized_validation_targets = validation_data['rank_target'] 
        self._feature_col_names = [column_name for column_name in self.validation_data.columns if 'feature' in column_name]
        self.example_predictions = example_preds
        self._rank_normalized_example_predictions = example_preds['rank_example_prediction'] # hardcoded
        self.starting_cols = validation_data.columns
    
    def score(self, df: pd.DataFrame)-> float:
        """
          # You should replace with lambda for speed
            utility to compute corr on a grouping of self._validation_data 
        """
        return _compute_corr(df['rank_target'], df['rank_prediction'])

    # suspect
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

        # BROKEN
    def compute_validation_corr(self)-> float:
        """
            pred: your predictions on the validation data.
            Compute your corr on the validation data.
            # need to call add_predictions_to_validation_df() before you call this or it will throw an derror
        """
  
        ranked_targets = self.rank_noramalize_series(self.validation_data['target'])
        ranked_preds = self.rank_noramalize_series(self.validation_data['prediction'])
        return self._compute_corr(ranked_targets, ranked_preds)


    def _compute_corr(self, a: pd.Series, b: pd.Series )->float:
        """
            Returns np.corrcoef on a and b. pass this only ranked correlations
        """
        return np.corrcoef(a, b)[0, 1] # not ranked 


    def compute_validation_std(self) -> float:
      """
          Returns the Standard Deviation of corr by era.
      """
      return self.create_per_era_grouper().std()

    def compute_validation_per_era_mean_corr(self)-> float:
      """
      Returns the mean corr by era.
      """
      return self.create_per_era_grouper().mean()

    def create_per_era_grouper(self) -> pd.Series:
      """
        Returns a series of era, Corr on validation targets for that era.
      """
      return self.validation_data.groupby("era").apply(lambda df: np.corrcoef(df['rank_target'], df['rank_prediction'])[0][1])

    def compute_validation_sharpe(self)-> float:
        """
            Computes your sharpe corr score on each era 
            sharpe = average corr per era / std dev of corr per era. 
        """
        per_era_corr_grouper = self.create_per_era_grouper()
        mean_per_era_corr = per_era_corr_grouper.mean()
        std_per_era_corr = per_era_corr_grouper.std()
        return mean_per_era_corr / std_per_era_corr
        

    def compute_max_drawdown(self)-> float:
        """
            Copied from Numerai's example_model.py
            Max drawdown is the "largest cumulative decrease between any two eras in terms of validation correlation"
            In short this keeps a running total of corr between eras. Then it find the length of the largest decrease and returns that number. 
            Is an estimate of risk
        """
        validation_correlations = self.create_per_era_grouper() # this needs to be stored in the class variables do avoid doing it twice
        rolling_max = (validation_correlations + 1).cumprod().rolling(window=100, min_periods=1).max()                                                           
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -(rolling_max - daily_value).max()
        return max_drawdown


    def compute_feature_exposure(self)-> float:
        """
            The maximum corrilatiosn your predictions have with any single feature
            Copied from Numerai's example_model.py
        """
        # pred_valid_df = self.validation_data # unclear if the default to 
        # pred_valid_df['prediction'] = rank_noramalize_series(pred) # add prediction column

        # feature_names = [f for f in self.validation_data.columns if f.startswith("feature")]
        # feature_exposures = validation_data[feature_names].apply(lambda d: correlation(pred_valid_df['prediction'], d), axis=0) # axis =0 means by columns
        feature_exposures = [self._compute_corr(self.validation_data[col], self.validation_data['prediction']) for col in self._feature_col_names]
        # for col in self._feature_col_names:
        #   feature_exposure_for_col = self._compute_corr(self.validation_data[col], self.validation_data['prediction']) # does feature exposrue look at rank corr 
        #   feature_exposures.append(feature_exposure_for_col)
                                                             
        max_feature_exposure = np.max(np.abs(np.array(feature_exposures)))
        return max_feature_exposure

 ################################################################################
 # BROKEN and unused
    def compute_feature_neutral_corr_mean(self)-> float:
        """
            Broken, and this is worse that just looking at feature exposure
            The mean of your per era correlation after your predictions have been neutralized to all the features
            Copied from Numerai's example_model.py
        """
        pred_valid_df = self.validation_data
        print('there should be prediction column and a rank_normalized prediction column')
        print(pred_valid_df.columns)
        # print(pred_valid_df.shape) 

        pred_valid_df.loc[:, "neutralized_predictions"] = self.neutralize(pred_valid_df, ['prediction'],
                                            self._feature_col_names)['prediction'] 
        num_rows = pred_valid_df.shape[0]
        scores = self.validation_data.groupby("era").apply(lambda df:self._compute_corr(df['neutralized_predictions'], 
                                                                                        df['rank_prediction']))

        # I the rank normalize within the lambda
        # scores = pred_valid_df.groupby("era").apply(
        #          lambda df: self._compute_corr(self.rank_noramalize_series(df["neutral_sub"]), 
        #                                        self.validation_data['rank_target'])).mean()

        print('scores is :')
        print(scores)
        return np.mean(scores)

####################################################################################

    #suspect
    def neutralize(self, df, columns, by, proportion=1.0):
        """
            Copied as is from example_model.py
        """
        scores = df.loc[:, columns] # scores is a df of all rows and only the coluns in columsn in 
        exposures = df[by].values

        # constant column to make sure the series is completely neutral to exposures
        exposures = np.hstack(
            (exposures,
            np.asarray(np.mean(scores)) * np.ones(len(exposures)).reshape(-1, 1)))

        scores = scores - proportion * exposures.dot(
            np.linalg.pinv(exposures).dot(scores))
        return scores / scores.std()
    
    #suspect
    def compute_mmc_stats(self, pred:pd.Series) -> tuple:
        """
            Using example predictions as an estimate for the meta model, compute mmc stats
            Copied from example_model.py
            returns val_mmc_mean, corr_plus_mmc_sharpe, 

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
        #corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

        # print(
        #     f"MMC Mean: {val_mmc_mean}\n"
        #     f"Corr Plus MMC Sharpe:{corr_plus_mmc_sharpe}\n"
        #     f"Corr Plus MMC Diff:{corr_plus_mmc_sharpe_diff}"
        # )
        return  val_mmc_mean, corr_plus_mmc_sharpe, 

    #suspect
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

    
    def compute_corr_with_example_preds(self, tournament_pred:pd.DataFrame) -> float:
        """
            Compute the rank corrilation between your tournament_pred and the example predictions
            tournament_pred: pd.DataFrame must have 'rank_prediction' column
            WORKS 
        """
        return self._compute_corr(tournament_pred['rank_prediction'], self.example_predictions['rank_example_prediction'])
    

    def add_predictions_to_validation_df(self, tournament_preds:pd.DataFrame) -> None:
      """
        updates the self.validation_df with your prediction in tournament_predss
        tournament_df: pd.DataFrame Your predictions for this round.
        Must have index = 'id'
        Must have column 'prediction' 
      """
      valid_ids = self.validation_data.index # get all the ids with the valaidatino data
      preds_on_valid_data = tournament_preds.loc[valid_ids,:] # subset on the validation data
      self.validation_data['prediction'] = preds_on_valid_data['prediction']
      rows = self.validation_data.shape[0] # number of rows
      self.validation_data['rank_prediction'] = self.validation_data['prediction'].rank(method='first')- 0.5 / rows
      return
    
    def reset_validation_df(self)-> None:
      """
          Remove the columns added to compute the scores
      """
      self.validation_data = self.validation_data.loc[:,self.starting_cols]


    def compute_numerai_diagnostics(self, tournament_preds: pd.DataFrame):
      """

          this is the main method that you call on your validation predictions
          my_score_calculator = ScoreCalculator()
          scores =my_score_calculator.compute_numerai_diagnostics(my_model.predict(validation_data[features]))
          print(scores)

          Return a dataframe that is equivalent to the diagnostics tab on numerai
          preds: A dataframe of your model's prediction accross the entire live tournament data for this round.
            Must have index='id'
                      columns = 'prediction', 'rank_prediction'
      """
      self.add_predictions_to_validation_df(tournament_preds)
      diagnostics_df = pd.DataFrame()
      diagnostics_df['valid_sharpe'] = [self.compute_validation_sharpe()]
      diagnostics_df['avg_valid_corr'] = [self.compute_validation_per_era_mean_corr()]
      diagnostics_df['valid_corr'] = [self.compute_validation_corr()] # wrong
      diagnostics_df['valid_std_dev'] = [self.compute_validation_std()]
      diagnostics_df['feature_exposure'] = [self.compute_feature_exposure()]
      diagnostics_df['max_drawdown'] = [self.compute_max_drawdown()]
      diagnostics_df['corr_with_example_preds '] = [self.compute_corr_with_example_preds(tournament_preds)]

      #diagnostics_df['valid_FNC'] = [self.compute_feature_neutral_corr_mean()] # hard copied as is from example_model.py
      #val_mmc_mean, corr_plus_mmc_sharpe = self.compute_mmc_stats(tournament_preds) # Hard copied as is from example_model.py
      #diagnostics_df['corr_plus_MMC_sharpe'] = [corr_plus_mmc_sharpe]
      #diagnostics_df['MMC_mean '] = [val_mmc_mean]
      self.reset_validation_df()
      
      return diagnostics_df.round(4)



class NumeraiDataLoader:
    """
      Pings and cleans the raw data from Numerai. Each method returns a DataFrame that is optimized for saving memory space.

      Much of the code is copy pasted. There is a lot of refactoring you can do to improve this method.
    """
    def ping_validation_data(self) -> pd.DataFrame:
        """
        Ping Numerai to create get the live tournament data and extact all the validation data.
        Runtime ~ 1.5 minutes.

        Adapted from : https://www.kaggle.com/code1110/numerai-tournament | May 3, 2021
        """
        tournament_data_url = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
        tournament_df = pd.read_csv(tournament_data_url, index_col=0)
        valid_df = tournament_df[tournament_df["data_type"] == "validation"].reset_index(drop = True)
        feature_cols = valid_df.columns[valid_df.columns.str.startswith('feature')]

        map_floats_to_ints = {0.0 : 0, 0.25 : 1, 0.5 : 2, 0.75 : 3, 1.0 : 4}
        for col in feature_cols:
            valid_df[col] = valid_df[col].map(map_floats_to_ints).astype(np.uint8) # reduce space costs by casting features as ints

        valid_df["era"] = valid_df["era"].apply(lambda x: int(x[3:])) # strip the word 'era' from the era column
        valid_df.drop(columns=["data_type"], inplace=True)

        total_valid_rows = valid_df.shape[0]
        valid_df['rank_target'] = valid_df['target'].rank(method='first') / total_valid_rows
        # valid_df.set_index('id', inplace=True)

        return valid_df


    def ping_tournament_data(self) -> pd.DataFrame: # Broken
        """
        Returns a Dataframe of this round, live tournament data. Run time ~ 5 minutes

        Adapted from : https://www.kaggle.com/code1110/numerai-tournament | May 3, 2021
        """
        tournament_data_url = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz'
        valid_df = pd.read_csv(tournament_data_url, index_col=0)
        feature_cols = valid_df.columns[valid_df.columns.str.startswith('feature')]
        
        map_floats_to_ints = {0.0 : 0, 0.25 : 1, 0.5 : 2, 0.75 : 3, 1.0 : 4}
        import traceback
        for col in feature_cols:
            valid_df[col] = valid_df[col].map(map_floats_to_ints).astype(np.uint8) # reduce space costs by casting features as ints
            
        try:              
            valid_df["era"] = valid_df["era"].apply(lambda x: int(x[3:])) # strip the word 'era' from the era column and cast as an int
        except:
            traceback.print_exc()

        valid_df.drop(columns=['data_type','target'], inplace=True)
        # valid_df.set_index('id', inplace=True)

        return valid_df


    def ping_example_predictions(self)-> pd.DataFrame:
      """
        Create a dataframe of id, Prediction, rank_prediction. Run time is ~2 second

        id              : The unique identifier for a row in the tournament data provided by Numerai
        prediction      : A float (0,1) that the example model predicts for that row. 
        rank_prediction : 'prediction' after it is rank normalized.

        Example : 

         	                  prediction  rank_prediction
        id		
        n0003aa52cab36c2	0.49	0.097334
        n000920ed083903f	0.49	0.097335
        n0038e640522c4a6	0.53	0.969455
        n004ac94a87dc54b	0.51	0.656894
        n0052fe97ea0c05f	0.50	0.332613
      """
      example_predictions_url = "https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_example_predictions_data.csv.xz"
      example_preds =  pd.read_csv(example_predictions_url, index_col=0) # defaults the index to be the 0th column
      total_example_prediction_rows = example_preds.shape[0]
      example_preds['rank_example_prediction'] = example_preds['prediction'].rank(method='first') / total_example_prediction_rows
      return example_preds


    def ping_training_data(self) -> pd.DataFrame:
        """
            Get the live training Data from numerai. Adds a Rank_target column to make the score calc faster.
            Runtime  1.5 minutes
        """
        training_data_url = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz'
        train_df = pd.read_csv(training_data_url)
        feature_cols = train_df.columns[train_df.columns.str.startswith('feature')]

        map_floats_to_ints = {0.0 : 0, 0.25 : 1, 0.5 : 2, 0.75 : 3, 1.0 : 4}
        for col in feature_cols:
            train_df[col] = train_df[col].map(map_floats_to_ints).astype(np.uint8) # reduce space costs by casting features as ints

        train_df["era"] = train_df["era"].apply(lambda x: int(x[3:])) # strip the word 'era' from the era column
        # train_df.drop(columns=["data_type"], inplace=True)

        total_rows = train_df.shape[0]
        train_df['rank_target'] = train_df['target'].rank(method='first') / total_rows
        train_df.set_index('id', inplace=True)

        return train_df

