import numpy as np

from scipy.stats import norm

from finmarkets.utils import OptionType

class VasicekModel:
    """
    A class to represent the Vasicek model

    Params:
    -------
    k: float
       k paramter for Vasicek model
    theta: float
        theta paramter for Vasicek model
    sigma: float
        sigma paramter for Vasicek model
    """
    def __init__(self, k, theta, sigma):
        self.theta = theta
        self.k = k
        self.sigma = sigma
        
    def r(self, r0, T, steps):
        """
        Evolves the short rate
        
        Params:
        -------
        r0: float
           initial rate value
        T: float
           time interval
        steps: int
           simulation steps
        """
        r = np.zeros(shape=(steps,))
        dt = T/steps
        r[0] = r0
        for i in range(1, steps):
            r[i] = r[i-1] + self.k*(self.theta-r[i-1])*dt +\
                   self.sigma*np.random.normal()*np.sqrt(dt)
        return r
            
    def B(self, t, T):
        return 1/self.k*(1-np.exp(-self.k*(T-t)))

    def A(self, t, T):
        return np.exp((self.theta-self.sigma**2/(2*self.k**2))*\
               (self.B(t, T)-T+t)-self.sigma**2/(4*self.k)*self.B(t, T)**2)
    
    def ZCB(self, r, t, T):
        """
        Compute the zero coupon bond price according to Vasicek model
        
        Params:
        -------
        r: float 
            initial short rate
        t: float
            today
        T: float
            maturity
        """
        return self.A(t, T)*np.exp(-self.B(t, T)*r)

    def ZBO(self, r, K, t, T, S, option_type=OptionType.Call):
        sigma_p = self.sigma*np.sqrt((1-np.exp(-2*self.k*(T-t)))/(2*self.k))*\
                  self.B(T, S)
        h = 1/sigma_p*np.log((self.ZCB(r, t, S))/(self.ZCB(r, t, T)*K))+sigma_p/2
        arg1 = option_type*h
        arg2 = option_type*(h-sigma_p)
        return option_type*(self.ZCB(r, t, S)*norm.cdf(arg1)-\
               K*self.ZCB(r, t, T)*norm.cdf(arg2))
    
