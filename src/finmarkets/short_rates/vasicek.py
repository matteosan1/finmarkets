import numpy as np

from finmarkets import maturity_from_str

class VasicekModel:
    """
    A class to represent the Vasicek model

    Params:
    -------
    k: float
       k paramter fo Vasicek model
    theta: float
        theta paramter fo Vasicek model
    sigma: float
        sigma paramter fo Vasicek model
    """
    def __init__(self, k, theta, sigma):
        self.k = k
        self.theta = theta
        self.sigma = sigma
        
    def r_generator(self, r0, dates, seed=1):
        """
        Evolves the short rate
        
        Params:
        -------
        r0: float
           initial rate value
        dates: list(datetime.date)
           the list of dates at which r has to be generated
        seed: int
           seed for the simulation (default=1)
        """
        if len(dates) < 2:
            print ("You need to pass at least two dates")
            return None
        np.random.seed(seed)
        dt = (dates[1] - dates[0]).days/365
        m = len(dates)
        r = np.zeros(shape=(m,))
        r[0] = r0
        for i in range(1, m):
            r[i] = r[i-1] + self.k*(self.theta - r[i-1])*dt + self.sigma*np.random.normal()*np.sqrt(dt)
        return r

    def _A(self, T):
        return ((self._B(T)-T)*(self.k**2*self.theta-self.sigma**2/2)/self.k**2) \
            - (self.sigma**2*self._B(T))/(4*self.k)
    
    def _B(self, T):
        return (1-np.exp(-self.k*T))/self.k
    
    def ZCB(self, T, r0):
        """
        Compute the zero coupon bond price according to Vasicek model
        
        Params:
        -------
        T: str
            Maturity of the bond
        r0: float
            Initial value of the rate
        """
        T = maturity_from_str(T, "y")
        return np.exp(self._A(T)-r0*self._B(T))


