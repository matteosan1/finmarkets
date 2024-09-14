import numpy as np, pandas as pd

from scipy.optimize import minimize

class Portfolio:
    def __init__(self, data):
        self.data = data
        self.daily_returns = df.dropna().pct_change()
        self.returns = self.daily_returns.mean()*252
        self.cov = self.daily_returns.cov()*252
        self.n = len(self.data.columns)
        self.w = np.ones(shape=(self.n,))*(1/self.n)

    def Pret(self):
        return self.w.dot(self.returns)

    def Pvar(self):
        return self.w.T.dot(self.cov.dot(self.w))
        
def sum_weights(w):
    return np.sum(w) - 1

def min_risk(w, cov):
    return w.T.dot(cov.dot(w))

def target_return(w, rets, target_return):
    return (rets.dot(w) - target_return)
    
