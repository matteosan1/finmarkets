import numpy as np

from scipy.stats import norm
from enum import IntEnum

OptionType = IntEnum("OptionType", {"Call":1, "Put":-1})

def BS(St, K, r, sigma, ttm, option_type):
    if type(ttm) == list:
        ttm = np.array([t for t in ttm])
    else:
        ttm = ttm

    return (option_type*St*norm.cdf(option_type*d_plus(St, K, r, sigma, ttm)) - option_type*K*np.exp(-r*(ttm))*norm.cdf(option_type*d_minus(St, K, r, sigma, ttm)))

def BSShifted(St, K, shift, r, sigma, ttm, option_type):
    K_shifted = K + shift
    St_shifted = St + shift
    return BS(St_shifted, K_shifted, r, sigma, ttm, option_type)

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
    ttm: float or list(float)
        time to maturity
    """
    if type(ttm) == list:
        ttm = np.array([t for t in ttm])
    else:
        ttm = ttm
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
    ttm: float or list(float)
        time to maturity
    """
    if type(ttm) == list:
        ttm = np.array([t for t in ttm])
    else:
        ttm = ttm
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

