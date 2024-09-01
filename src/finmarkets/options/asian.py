import numpy as np

from scipy.stats.mstats import gmean
from scipy.stats import norm

from finmarkets.utils import OptionType

class AsianOption:
    """
    A class to represents Asian Options

    Params:
    -------
    S0: float
        underlying spot price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: float or list(float)
        time to maturity
    otype: OptionType enum
        Call, Put (default value Call)
    """
    def __init__(self, S0, K, r, sigma, ttm, otype=OptionType.Call):
        self.S0 = S0
        self.K = K
        if type(ttm) == list:
            self.ttm = np.array([t for t in ttm])
        else:
            self.ttm = ttm
        self.sigma = sigma
        self.r = r
        self.otype = otype

    def geometric_price(self, timesteps=252):
        """
        Compute geometric asian option price with analytical formula

        Params:
        -------
        timesteps: int
           number of intervals to compute geometric mean 
        """
        adj_sigma = self.sigma*np.sqrt((2*timesteps+1)/(6*(timesteps+1)))
        rho = 0.5*(self.r-(self.sigma**2)*0.5 + adj_sigma**2)
        d1 = (np.log(self.S0/self.K)+(rho+0.5*adj_sigma**2)*self.ttm)/(adj_sigma*np.sqrt(self.ttm))
        d2 = (np.log(self.S0/self.K)+(rho-0.5*adj_sigma**2)*self.ttm)/(adj_sigma*np.sqrt(self.ttm))
        if self.otype == OptionType.Call:
            return np.exp(-self.r*self.ttm)*(self.S0*np.exp(rho*self.ttm)*norm.cdf(d1)-self.K*norm.cdf(d2))
        else:
            return np.exp(-self.r*self.ttm)*(self.K*norm.cdf(-d2)-self.S0*np.exp(rho*self.ttm)*norm.cdf(-d1))

    def path(self, n, timesteps, anti=False, seed=10000):
        """
        Compute paths for underlying price assuming log-normal evolution

        Params:
        -------
        n: int
            number of paths to simulate
        timesteps: int
            number of time steps to be used in the simulation
        anti: Boolean
            flag to invert sign of Brownian motion realizations
        seed: int
            pseudorandom number seed
        """  
        np.random.seed(seed)
        dt = self.ttm/timesteps
  
        S = np.zeros(shape=(timesteps, n))
        S[0] = self.S0
        w = norm.rvs(size=(timesteps-1)*n).reshape(((timesteps-1), n))
        for i in range(0, timesteps-1):
            S[i+1] = S[i] * (1 + self.r*dt  + self.sigma*np.sqrt(dt)*w[i])

        if anti:
            S1 = np.zeros(shape=(timesteps, n))
            S1[0] = self.S0
            w = -w
            for i in range(0, timesteps-1):
                S1[i+1] = S1[i] * (1 + self.r*dt  + self.sigma*np.sqrt(dt)*w[i])

        if anti:
            return S, S1
        else:
            return S

    def aritmethic_payoff(self, S):
        """
        Computes option payoff for aritmethic type

        Params:
        -------
        S: np.array
            array with underlying price realizations
        """
        if self.otype == OptionType.Call:
            return np.maximum(0, np.mean(S, axis=0)-self.K)
        else:
            return np.maximum(0, self.K - np.mean(S, axis=0))

    def geometric_payoff(self, S):
        """
        Computes option payoff for geometric type

        Params:
        -------
        S: np.array
            array with underlying price realizations
        """
        if self.otype == OptionType.Call:
            return np.maximum(0, gmean(S, axis=0)-self.K)
        else:
            return np.maximum(0, self.K-gmean(S, axis=0))

    def simulate_naive(self, n, timesteps=252, seed=10000):
        """
        Computes asian option price through naive MC

        Params:
        -------
        n: int
            number of simulations
        timesteps: int
            time steps to be used in the simulations (default 252)
        seed: int
            pseudorandom number generator seed (default 10000)
        """
        S = self.path(n, timesteps, seed=seed)
        payoffs = np.exp(-self.r*self.ttm) * self.aritmethic_payoff(S)
        price = np.mean(payoffs)
        err = np.std(payoffs)/np.sqrt(n)
        return price, err

    def simulate_antithetic(self, n, timesteps=252, seed=10000):
        """
        Computes asian option price through antithetic technique

        Params:
        -------
        n: int
            number of simulations
        timesteps: int
            time steps to be used in the simulations (default 252)
        seed: int
            pseudorandom number generator seed (default 10000)
        """
        S1, S2 = self.path(n, timesteps, anti=True, seed=seed)
        S = np.exp(-self.r*self.ttm) * (self.aritmethic_payoff(S1) + self.aritmethic_payoff(S2))/2
        price = np.mean(S)
        err = np.std(S)/np.sqrt(n)
        return price, err
    
    def simulate_control_variate(self, n, timesteps=252, seed=10000):
        """
        Computes asian option price through control variate technique

        Params:
        -------
        n: int
            number of simulations
        timesteps: int
            time steps to be used in the simulations (default 252)
        seed: int
            pseudorandom number generator seed (default 10000)
        """
        S = self.path(n, timesteps)
        X = np.exp(-self.r*self.ttm) * self.aritmethic_payoff(S)
        Y = np.exp(-self.r*self.ttm) * self.geometric_payoff(S)
        c = -np.cov(X,Y)[0, 1]/np.var(Y)
        mu = self.geometric_price(timesteps)
        Z = X + c*(Y-mu)
        price = np.mean(Z)
        err = (np.std(Z)/np.sqrt(n))
        return price, err
    
