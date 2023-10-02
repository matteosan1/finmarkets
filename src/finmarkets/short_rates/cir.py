import numpy as np
from numpy.random import seed, normal

class CIRModel:
    """
    A class to represent the CIR model

    Params:
    -------
    k: float
       k paramter fo CIR model
    theta: float
        theta paramter fo CIR model
    sigma: float
        sigma paramter fo CIR model
    """
    def __init__(self, k, theta, sigma):
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.gamma = np.sqrt(self.k**2 + 2*self.sigma**2)
        self.setSeed()

    def setSeed(self, aseed=1):
        """
        Sets the seed of the random number generator
        
        Params:
        -------
        aseed: int
            The seed to set
        """
        seed(aseed)
        
    def r_generator(self, r0, T, m=100):
        """
        Evolves the short rate
        
        Params:
        -------
        r0: float
           Initial rate value
        T: float
           The length of the rate evolution
        m: int
           Number of steps of the evolution
        """
        dt = T/m
        r = np.zeros(shape=(m,))
        r[0] = r0
        for i in range(1, m):
            r[i] = r[i-1] + self.k*(self.theta - r[i-1])*dt \
                + self.sigma*np.random.normal()*np.sqrt(dt*r[i-1])
        return r

    def _B(self, T):
        c = np.exp(self.gamma*T) - 1
        return 2*c/((self.gamma + self.k)*c + 2*self.gamma)

    def _A(self, T):
        c = np.exp(self.gamma*T) - 1
        num = 2*self.gamma*np.exp((self.k+self.gamma)*T/2)
        den = (self.gamma + self.k)*c + 2*self.gamma
        return np.power(num/den, 2*self.k*self.theta/self.sigma**2)
    
    def ZCB(self, T, r0):
        """
        Compute the zero coupon bond price according to CIR model
        
        Params:
        -------
        T: float
            Maturity of the bond
        r0: float
            Initial value of the rate
        """
        return self._A(T)*np.exp(-r0*self._B(T))
