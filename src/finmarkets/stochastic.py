import numpy as np

from numpy.random import normal, seed
from scipy.stats import lognorm

def BM(mu, sigma, x0, T, tsteps, N):
    """
    Simualate Brownian motion realizations

    Params:
    -------
    mu: float
        drift of the process
    sigma: float 
        diffusion coefficient
    x0: float
        initial value
    T: float 
        length of the simulation
    tsteps: int
        number of steps to simulate
    N: int
        number of simulations
    """
    x = np.zeros(shape=(tsteps, N))
    x[0, :] = x0
    epsilon = np.random.normal(size=(tsteps-1, N))
    x[1:, :] = mu*dt + sigma*epsilon*np.sqrt(T/tsteps)
    return np.cumsum(x, axis=0)
  
def GBM(mu, sigma, X0, T, tsteps, N):
    """
    Simualate Geometric Brownian motion realizations

    Params:
    -------
    mu: float
        drift of the process
    sigma: float 
        diffusion coefficient
    X0: float
        initial value
    T: float 
        length of the simulation
    tsteps: int
        number of steps to simulate
    N: int
        number of simulations
    """
    X = np.zeros(shape=(tsteps, N))
    dt = T/tsteps
    X[0, :] = X0
    epsilon = np.random.normal(size=(tsteps-1, N))
    X[1:, :] = np.exp((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*epsilon)
    return np.cumprod(X, axis=0)

def GBMShifted(mu, sigma, shift, X0, T, tsteps, N):
    """
    Simualate a shifted Geometric Brownian motion realizations

    Params:
    -------
    mu: float
        drift of the process
    sigma: float 
        diffusion coefficient
    shift: float
        shift to apply
    X0: float
        initial value
    T: float 
        length of the simulation
    tsteps: int
        number of steps to simulate
    N: int
        number of simulations
    """
    X0_shifted = X0 + shift
    if (X0_shifted < 0.0):
        raise ValueError('Shift is too small !')

    X_shifted = GBM(mu, sigma, X0_shifted, T, tsteps, N)
    return X_shifted - shift

def shifted_lognormal(x, mu, sigma, dt, shift):
    return lognorm.pdf(x+shift, scale = np.exp(np.log(mu+shift) + (-0.5*sigma**2)*dt), s=np.sqrt(dt)*sigma)


class LogNormalEv:
    """
    A class to model a LogNormal Stochastic process

    Params:
    -------
    S0: float
        Initial value of the random variable
    mu: float
        Drift of the random variable
    sigma: float
        Volatitlity of the random variabe
    T: float
        Length of the simulation
    steps: int
        Number of steps to simulate
    """
    def __init__(self, S0, mu, sigma, T, steps=100):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.steps = steps+1
        self.dT = self.T/self.steps
        self.setSeed()
        
    def setSeed(self, aseed=1):
        """
        Set the seed of the random number generator

        Params:
        -------
        aseed: int
            The seed to set
        """
        seed(aseed)
        
    def simulate(self):
        """
        Simulates the evolution of the random variable
        """
        S = np.ones(shape=(self.steps,))*self.S0
        for i in range(1, self.steps):
            S[i] = S[i-1] * np.exp((self.mu - 0.5 * self.sigma**2)*self.dT +
                                   self.sigma * np.sqrt(self.dT) * normal())
        return S
