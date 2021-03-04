import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import intro_numerai_api as local_api


# I look at the csv file I created eariler to look at some regression and basic exploritaiy stats on the profitablity and other stats for the leader board. 


# in theroy you can do this periodily to see the piciture over time

# I dont' have a clean way of getting the data right now I just copy and past it to notepad. 


def create_scatter_plot(df, x_name='rolling_score_rep', y_name ='nmrStaked'):
    """
    Scatter Plot of Corrilation vs Stake

    """
    x = df[x_name]
    y = df[y_name]
    plt.scatter(x,y, s=.5)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def create_histogram(df, col='rolling_score_rep', bins =10):
    x=df[col]
    plt.hist(x,bins)
    plt.xlabel(col)
    plt.show()


def custom_describe(df, col):
    percents = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    print(df[col].describe(percentiles=percents))


def main():
    df = local_api.load_leaderboard()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    create_scatter_plot(df)
    
    

main()

