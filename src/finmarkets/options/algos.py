import numpy as np

from finmarkets.dates import dt_from_str
from finmarkets.stochastic import GBM
from .vanilla import OptionType

def longstaff_schwartz(S0, r, sigma, K, tsteps, T, N, side=OptionType.Call):
    #T = dt_from_str(maturity)
    dt = T / tsteps
    df = np.exp(-r*dt)    
    S = GBM(r, sigma, S0, T, tsteps, N)
   
    h = np.maximum(side*(S-K), 0)

    V = h[-1]
    for t in range(tsteps-1, 0, -1):
        regr = np.polyfit(S[t], V*df, 5)
        C  = np.polyval(regr, S[t])
        V  = np.where(h[t]>C, h[t], V*df)
    return df*np.sum(V)/N
