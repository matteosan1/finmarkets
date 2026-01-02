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

class PiecewisePoissonProcess(rv_continuous):
    """
    A class to describe a piecewise poisson process.
    
    Params:
    -------
    lambdas: np.array(float)
        lambda parameters of the distribution
    times: np.array(float)
        times defining the piecewise intervals
    """
    def __init__(self, lambdas, times):
        super().__init__(a=0.0)
        self.lambdas = np.array(lambdas)
        self.times = np.array(times)

    def _cumulative_intensity(self, x):
        """
        Calculates the cumulative intensity

        Params:
        -------
        x: float or numpy.array
            values where to compute the CDF
        """
        x = np.atleast_1d(x)
        cum_int = np.zeros_like(x, dtype=float)

        for i, val in enumerate(x):
            if val <= 0:
                continue

            total = 0.0
            prev_t = 0.0
            for lam, t in zip(self.lambdas, self.times):
                if val <= t:
                    total += lam * (val - prev_t)
                    prev_t = val
                    break
                total += lam * (t - prev_t)
                prev_t = t

            if val > prev_t:
                total += self.lambdas[-1] * (val - prev_t)

            cum_int[i] = total
        return cum_int

    def _hazard_rate(self, x):
        """
        Returns the proper value of lambda given a time

        Params:
        -------
        x: float or numpy.array
            values at which to return the lambdas
        """
        x = np.atleast_1d(x)
        conditions = [x <= t for t in self.times]
        return np.piecewise(x, conditions, list(self.lambdas) \
               + [self.lambdas[-1]])

    def _cdf(self, x):
        """
        Reimplements the same method from parent class

        Params:
        -------
        x: float or numpy.array
            values where to compute the distribution CDF
        """
        return 1 - np.exp(-self.cumulative_intensity_vectorized(x))

    def _pdf(self, x):
        """
        Reimplements the same method from parent class

        Params:
        -------
        x: float or numpy.array
            values where to compute the distribution PDF
        """
        return self._hazard_rate(x) \
               * np.exp(-self.cumulative_intensity_vectorized(x))

    def cumulative_intensity_vectorized(self, x):
        """
        Helper function to return CDF

        Params:
        -------
        x: float or numpy.array
            values where to compute the CDF
        """
        if np.isscalar(x):
            return self._cumulative_intensity([x])[0]
        return self._cumulative_intensity(x)

    def survival_prob(self, x):
        """
        Calculates the survival probability as 1-Pdef

        Params:
        -------
        x: float or numpy.array
            values where to compute the survival probabilities
        """
        return 1 - self._cdf(x)
