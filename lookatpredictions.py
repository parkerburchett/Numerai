import csv
import pandas as pd
import matplotlib.pyplot as plt

preds = pd.read_csv('myPredictions.csv')

def lookatpreds(preds):

    print(preds.count())
    print(preds.prediction.describe())

    plt.hist(preds.prediction)
    plt.show()

example_preds = pd.read_csv('example_predictions.csv')

lookatpreds(preds)