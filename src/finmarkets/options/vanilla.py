import numpy as np

from scipy.stats.mstats import gmean
from scipy.stats import norm

from finmarkets import maturity_from_str

def call(St, K, r, sigma, ttm):
    """
    Compute call price through Black-Scholes formula

    Params:
    -------
    St: float
        underlying spot price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: str or list(str)
        time to maturity
    """
    if type(ttm) == list:
        ttm = np.array([maturity_from_str(t, "y") for t in ttm])
    else:
        ttm = maturity_from_str(ttm, "y")
    return (St*norm.cdf(d_plus(St, K, r, sigma, ttm)) -
            K*np.exp(-r*(ttm))*norm.cdf(d_minus(St, K, r, sigma, ttm)))

def put(St, K, r, sigma, ttm):
    """
    Computes put price through Black-Scholes formula

    Params:
    -------
    St: float
        underlying spot price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: str
        time to maturity
    """
    if type(ttm) == list:
        ttm = np.array([maturity_from_str(t, "y") for t in ttm])
    else:
        ttm = maturity_from_str(ttm, "y")
    return (K*np.exp(-r*(ttm))*norm.cdf(-d_minus(St, K, r, sigma, ttm)) -
            St*norm.cdf(-d_plus(St, K, r, sigma, ttm)))
    
def d_plus(St, K, r, sigma, ttm):
    """
    Computes d_plus coefficient for Black-Scholes formula

    Params:
    -------
    St: float
        underlying price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: float
        time to maturity in years
    """
    num = np.log(St/K) + (r + 0.5*sigma**2)*(ttm)
    den = sigma*np.sqrt(ttm)
    return num/den

def d_minus(St, K, r, sigma, ttm):
    """
    Computes d_minus coefficient for Black-Scholes formula

    Params:
    -------
    St: float
        underlying price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: float
        time to maturity in years
    """
    return d_plus(St, K, r, sigma, ttm) - sigma*np.sqrt(ttm)

