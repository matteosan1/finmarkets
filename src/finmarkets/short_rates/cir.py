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
                
    def r(self, r0, T, steps):
        """
        Evolves the short rate according to the CIR model
        
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
        r = np.zeros(steps)
        r[0] = r0 
        for t in range(1, steps):
            dr = r[t-1]
            r[t] = dr + self.k*(self.theta-dr)*dt + self.sigma*np.sqrt(dt)*np.sqrt(max(0, dr))*np.random.normal()
        return r    

    def _B(self, T):
        c = np.exp(self.gamma*T) - 1
        return 2*c/((self.gamma + self.k)*c + 2*self.gamma)

    def _A(self, T):
        c = np.exp(self.gamma*T) - 1
        num = 2*self.gamma*np.exp((self.k+self.gamma)*T/2)
        den = (self.gamma + self.k)*c + 2*self.gamma
        return np.power(num/den, 2*self.k*self.theta/self.sigma**2)

    def ZCB(self, r0, t, T):
        """
        Computes the price of a zero-coupon bond with maturity T using the CIR short rate model.

        Params:
        -------
        r0: float
            initial short rate.
        t: float 
            current time.
        T: float    
            maturity of the bond.
        """
        return self._A(T)*np.exp(-r0*self._B(T))

