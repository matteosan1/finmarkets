import numpy as np

from finmarkets import maturity_from_str

class HullWhiteModel:
    """
    A class to represent the Hull-White model

    Params:
    -------
    a: float
        paramter for Hull-White model
    sigma: float
        sigma paramter for Hull-White model
    """
    def __init__(self, a, sigma):
        self.a = a
        self.sigma = sigma
        
    def zero_bond(self, t, T):
        """
        Calculates the price of a zero-coupon bond at time t maturing at time T.
        """
        B = np.exp(-self.a*(T - t) + (self.sigma**2/(4*self.a**3))*(1 - np.exp(-2*self.a*(T - t)))*(1 - 2*self.a*(T - t)))
        return B
