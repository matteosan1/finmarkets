import numpy as np
from numpy.random import normal, seed

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
