import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# I look at the csv file I created eariler to look at some regression and basic exploritaiy stats on the profitablity and other stats for the leader board. 


# in theroy you can do this periodily to see the piciture over time

# I dont' have a clean way of getting the data right now I just copy and past it to notepad. 


def create_scatter_plot(df):
    """
    Scatter Plot of Corrilation vs Stake

    """
    x = df[' corr']
    y = df[' stake']
    plt.scatter(x,y, s=.5)
    plt.xlabel('Corr')
    plt.ylabel('Stake')
    plt.show()


def create_histogram(df, col='corr', bins =20):
    x = df[col] # you will need to find a way to filter out the zeros. right now I just can't  think of that
    plt.hist(x,bins)
    plt.xlabel(variable)
    plt.show()



def custom_describe(df, col):
    percents = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    print(df[col].describe(percentiles=percents))

def main():
    df = pd.read_csv(r'C:\Users\parke\Documents\GitHub\Numerai\finished_cleaned_users.csv')

    #print(df.head())
    print(df.columns)
    #create_histogram(df, 'roi_1_year', bins =100)

    # custom_describe(df, 'corr')
    # custom_describe(df,'roi_1_day')
    #print(df['roi_1_day'].describe())
    

main()

