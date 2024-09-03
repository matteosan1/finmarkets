import numpy as np

from scipy.stats import norm, rv_continuous

class PoissonProcess(rv_continuous):
    """
    A class to describe lambda * exp(-lambda*x) distributions, inherits from rv_continuous.
    
    Params:
    -------
    lambda: float
        lambda parameter of the distribution
    """
    def __init__(self, l):
        super().__init__()
        self.l = l

    def _cdf(self, x):
        """
        Reimplements the same method from parent class

        Params:
        -------
        x: float or numpy.array
            values where to compute the distribution CDF
        """
        x[x < 0] = 0
        return (1 - np.exp(-self.l*x))

    def _pdf(self, x):
        """
        Reimplements the same method from parent class

        Params:
        -------
        x: float or numpy.array
            values where to compute the distribution PDF
        """
        x[x < 0] = 0
        return self.l*np.exp(-self.l*x)

    def _ppf(self, x):
        """
        Reimplement the same method from parent class

        Params:
        -------
        x: float or numpy.array
            values where to compute the distribution PPF
        """
        return -np.log(1-x)/self.l

    
