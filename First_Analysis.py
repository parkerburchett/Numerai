import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import intro_numerai_api as local_api
import pingGraphQL
import statsmodels.api as sm

# I look at the csv file I created eariler to look at some regression and basic exploritaiy stats on the profitablity and other stats for the leader board.

# in theroy you can do this periodily to see the piciture over time

# I dont' have a clean way of getting the data right now I just copy and past it to notepad.


# you might want to show sevearl min stakes, as a range. eg create sevearl box plots all on one show
def create_box_plot(df, col='corrRep', min_stake=0):
    x = df[(df[col].notnull()) & (df['nmrStaked'] > min_stake)][col]
    plt.boxplot(x, vert=False)
    num_elements = x.count()
    plt.title(
        f'Distribution of {col} nmrStaked>{min_stake}\n Number of Elements:{num_elements}'
    )
    plt.xlabel(col)
    plt.show()


# good
def create_scatter_plot(df, x_name='corrRep', y_name='nmrStaked', min_stake=0):
    x = df[(df[x_name].notnull()) & (df['nmrStaked'] > min_stake)][x_name]
    y = df[(df[y_name].notnull()) & (df['nmrStaked'] > min_stake)][y_name]
    plt.scatter(x, y, s=.5)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    num_elements = x.count()
    plt.title(
        f'{y_name} v {x_name} when nmrStaked>{min_stake}\n Number of Elements:{num_elements}'
    )
    plt.show()


# good
def create_histogram(df, col='nmrStaked', bins=50, min_stake=0):
    x = df[(df[x_name].notnull()) & (df['nmrStaked'] > min_stake)][x_name]
    plt.hist(x, bins)
    plt.xlabel(col)
    num_elements = x.count()
    plt.title(
        f'Histogram of {col} when nmrStaked>{min_stake}\n Number of Elements:{num_elements}'
    )
    plt.show()


# good
def custom_describe(df, col='corrRep', min_stake=0):
    percents = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    x = df[(df[col].notnull()) & (df['nmrStaked'] > min_stake)][col]
    print(f'{col} when nmrStaked>{min_stake}\n')
    print(x.describe(percentiles=percents))


# good
def compute_single_regression(df,
                              independent_variable='corrRep',
                              dependent_variable='3M_returns',
                              min_stake=0):


    x = df[(df[dependent_variable].notnull())
           & (df['nmrStaked'] > min_stake)][independent_variable]
    y = df[(df[dependent_variable].notnull())
           & (df['nmrStaked'] > min_stake)][dependent_variable]
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())


# good
def compute_multiple_regression(
        df,
        indepenent_variables=['corrRep', 'fncRep', 'mmcRep', 'nmrStaked'],
        dependent_variable='3M_returns',
        min_stake=0):
    x = df[(df[dependent_variable].notnull())
           & (df['nmrStaked'] > min_stake)][indepenent_variables]
    y = df[(df[dependent_variable].notnull())
           & (df['nmrStaked'] > min_stake)][dependent_variable]
    model = sm.OLS(y, x)
    results = model.fit()
    print(results.summary())


def main():
    df = pingGraphQL.get_leaderboard()
    custom_describe(df, min_stake=100)


main()
