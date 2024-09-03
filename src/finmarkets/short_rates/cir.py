import numpy as np

class CIRModel:
    """
    A class to represent the CIR model

    Params:
    -------
    k: float
        k parameter of CIR model
    theta: float
        theta parameter of CIR model
    sigma: float
        sigma parameter of CIR model
    """
    def __init__(self, k, theta, sigma):
        self.k = k
        self.theta = theta
        self.sigma = sigma
                
    def r_generator(self, r0, T, steps):
        """
        Evolves the short rate
        
        Params:
        -------
        r0: float
           Initial rate value
        T: float
           Terminal time
        steps: int
           Number of steps to simulate
        """
        dt = T / steps
        r = np.zeros(steps+1)
        r[0] = r0 
        for t in range(1, steps+1):
            Z = np.random.randn()
            dr = r[t-1]
            r[t] = dr + self.k*(self.theta-dr)*dt + self.sigma*np.sqrt(dt)*np.sqrt(max(0, dr))*Z    
        return r    
        
    def ZCB(self, t, T, r0):
        """
        Computes the price of a zero-coupon bond with maturity T using the CIR short rate model.

        Params:
        -------
        t: float 
            Current time.
        T: float    
            Maturity of the bond.
        r0: float
            Initial short rate.
        """
        a = self.k*(self.theta - r0)
        b = 0.5*self.sigma**2

        return np.exp(-a/b*(1 - np.exp(-b*(T - t))) - 2*self.k*self.theta/self.sigma**2*(T - t + (1 - np.exp(-b*(T - t)))/b))

        