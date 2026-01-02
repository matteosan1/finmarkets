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
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def r(self, r0, N, T, dt, Z=None):
        """
        Evolves the short rate
        
        Params:
        -------
        r0: float
           initial rate value
        N: int
           number of simulations
        T: float
           maturity
        dt: float
            time interval
        Z: np.array
           precomputed brownian motion (optional)   
        """
        M = int(T/dt)
        r = np.zeros(shape=(N, M,))
        r[:, 0] = r0
        if Z is None:
            Z = np.random.normal(size=(N, M))
        for i in range(1, M):            
            r[:, i] = r[:, i-1] + self.a * (self.b - r[:, i-1]) * dt + self.sigma * np.sqrt(dt) * Z[:, i]
        return r

    def R(self, paths, T):
        """
        Convert short rate paths into yield curves
        
        Params:
        -------
        paths: np.array
           all the simulated paths
        T: float
            time to maturity
        """        
        num_steps = paths.shape[1]
        dt = T / (num_steps - 1) if num_steps > 1 else T
        time_diff = T - np.arange(num_steps) * dt
        R_t_T = np.zeros_like(paths)
        for i in range(R_t_T.shape[0]):            
            R_t_T[i, :-1] = (self.B(time_diff[:-1]) * paths[i, :-1] - np.log(self.A(time_diff[:-1]))) / time_diff[:-1]
    
        R_t_T[:, -1] = paths[:, -1]
        return R_t_T
    
    def B(self, delta_t):
        return (1-np.exp(-self.a*delta_t))/self.a

    def A(self, delta_t):
        return np.exp((self.b-self.sigma**2/(2*self.a**2))*(self.B(delta_t)-delta_t)-(self.sigma*self.B(delta_t))**2/(4*self.a))
    
    # def ZBO(self, K, t, T, S, option_type=OptionType.Call):
    #     sigma_p = self.sigma*np.sqrt((1-np.exp(-2*self.k*(T-t)))/(2*self.k))*\
    #               self.B(T, S)
    #     h = 1/sigma_p*np.log((self.ZCB(t, S))/(self.ZCB(t, T)*K))+sigma_p/2
    #     arg1 = option_type*h
    #     arg2 = option_type*(h-sigma_p)
    #     return option_type*(self.ZCB(t, S)*norm.cdf(arg1)-K*self.ZCB(t, T)*norm.cdf(arg2))
    
    def ZCB_analytical(self, r0, T):
        """
        Compute zero coupon bond prices
      
        Params:
        -------
        r0: float
            Spot rate
        T: float
            Maturity (years)
        """    
        price = self.A(T) * np.exp(-self.B(T) * r0)
        return price
    
    def ZCB(self, paths, T):        
        """
        Compute zero coupon bond prices
        
        Params:
        -------
        paths: np.array
           all the simulated paths
        T: float
            time to maturity
        """    
        num_steps = paths.shape[1]
        dt = T / (num_steps - 1) if num_steps > 1 else T
        time_diff = T - np.arange(num_steps) * dt
        
        P_t_T = np.zeros_like(paths)
        for i in range(P_t_T.shape[0]): 
            P_t_T[i, :-1] = self.A(time_diff[:-1]) * np.exp(-self.B(time_diff[:-1] * paths[i, :-1]))

        if num_steps > 0:
            P_t_T[:, -1] = 1.0
        
        return P_t_T
