import pandas as pd
import matplotlib.pyplot as plt
import numerapi
import inspect


def look_at_my_predictions():
    preds = pd.read_csv('myPredictions.csv')
    print(preds.count())
    print(preds.prediction.describe())
    plt.hist(preds.prediction)
    plt.show()

lines = inspect.getsource(numerapi)
print(lines)