import numpy as np


# source https://www.youtube.com/watch?v=50oyD_e8Vh0
def compute_sharpe(a):
    average_apr =np.average(a)
    risk_free  =.01
    annual_risk=np.std(a)
    return (average_apr - risk_free)/ annual_risk



s = [.1,.2,-.05,-.1,.01,.4,.1,.2]

print(compute_sharpe(s))