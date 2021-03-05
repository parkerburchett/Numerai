import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import intro_numerai_api as local_api
import pingGraphQL
import statsmodels.api as sm


# I look at the csv file I created eariler to look at some regression and basic exploritaiy stats on the profitablity and other stats for the leader board. 


# in theroy you can do this periodily to see the piciture over time

# I dont' have a clean way of getting the data right now I just copy and past it to notepad. 


def create_scatter_plot(df, x_name='corrRep', y_name ='nmrStaked'):
    """
    Scatter Plot of Corrilation vs Stake

    """
    x = df[x_name]
    y = df[y_name]
    plt.scatter(x,y, s=.5)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def create_histogram(df, col='nmrStaked', bins =50, log=True):
    x=df[col]
    plt.hist(x,bins, log=log)
    plt.xlabel(col)
    plt.show()


def custom_describe(df, col):
    percents = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    print(df[col].describe(percentiles=percents))


def compute_single_regression(df, independent_variable='corrRep', dependent_variable ='return_13Weeks'):
    local_df =df[df[dependent_variable].notnull()]
    print(local_df.columns)
    x = local_df[independent_variable]
    y = local_df[dependent_variable]
    model =sm.OLS(y,x)
    results = model.fit()
    print(results.summary())
    
def compute_multiple_regression(df, indepenent_variables= ['corrRep', 'fncRep', 'mmcRep', 'nmrStaked'], dependent_variable ='return_13Weeks'):
    local_df =df[df[dependent_variable].notnull()]
    x = local_df[indepenent_variables]
    y = local_df[dependent_variable]
    model =sm.OLS(y,x)
    results = model.fit()
    print(results.summary())


def main():
    df = pingGraphQL.custom_ping_leaderboad()
    print(df.columns)
    compute_multiple_regression(df)


main()

