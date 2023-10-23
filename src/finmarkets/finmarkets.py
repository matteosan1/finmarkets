import numpy as np, pickle

from scipy.stats import norm, rv_continuous, binom, multivariate_normal
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from datetime import date
from dateutil.relativedelta import relativedelta

from .dates import maturity_from_str, generate_dates

def saveObj(filename, obj):
    """
    Utility function to pickle any "finmarkets" object

    Params:
    -------
    filename: str
        filename of the pickled object
    obj: finmarkets object
        the object to pickle
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def loadObj(filename):
    """
    Utility function to unpickle any "finmarkets" object

    Params:
    -------
    filename: str
        filename of the object to unpickle
    """    
    with open(filename, "rb") as f:
        return pickle.load(f)

class DiscountCurve:
    """
    A class to represent discount curves

    Attributes:
    -----------
    obs_date: datetime.date
        observation date.
    pillar_dates: list(datetime.date)
        pillars dates of the discount curve
    discount_factors: list(float)
        actual discount factors
    """
    def __init__(self, obs_date, pillar_dates, discount_factors):
        discount_factors = np.array(discount_factors)
        if obs_date not in pillar_dates:
            pillar_dates = [obs_date] + pillar_dates
            discount_factors = np.insert(discount_factors, 0, 1)
        self.pillars = [p.toordinal() for p in pillar_dates] 
        self.log_discount_factors = np.log(discount_factors)
        self.interpolator = interp1d(self.pillars, self.log_discount_factors)
        
    def df(self, adate):
        """
        Gets interpolated discount factor at `adate`

        Params:
        -------
        adate: datetime.date
            actual date at which we would like the interpolated discount factor
        """
        d = adate.toordinal()
        if d < self.pillars[0] or d > self.pillars[-1]:
            print (f"Cannot extrapolate discount factors (date: {adate}).")
            return None
        return np.exp(self.interpolator(d))
    
    def annualized_yield(self, d):
        """
        Computes the annualized yield at a given date

        Params:
        -------
        d: datetime.date
            actual date at which calculate the yield
        """
        return -np.log(self.df(d))/((d-self.obs_date).days/365) 

#def makeDCFromDataFrame(df, obs_date, pillar_col='months', df_col='dfs'):
#    """
#    makeDCFromDataFrame - utility to create a DiscountCurve object from a pandas.DataFrame.
#    
#    Params:
#    -------
#    df: pandas.DataFrame
#        Input pandas.DataFrame.
#    obs_date: datetime.date
#        Observation date.
#    pillar_col: str
#        Name of the pillar column in df, default is 'months'.
#    df_col: str
#        Name of discount factors column in df, default is 'dfs'.
#    """
#    pillars = [today + relativedelta(months=i) for i in df[pillar_col]]
#    dfs = df[df_col]
#    return DiscountCurve(obs_date, pillars, dfs)

class ForwardRateCurve:
    """
    A class to represent a forward rate curve

    Attributes:
    -----------
    obs_date: datetime.date
        observation date.
    pillar_dates: list(datetime.date)
        pillar dates of the forward rate curve
    rates: list(float)
        rates of the forward curve
    """
    def __init__(self, obs_date, pillars, rates):
        self.obs_date = obs_date
        self.pillars = [(p-obs_date).days/365 for p in pillars]
        self.rates = rates
        self.interpolator = interp1d(self.pillars, self.rates)
        
    def interp_rate(self, adate):
        """
        Find the rate at time d
        
        Params:
        -------
        adate : datetime.date
            date of the interpolated rate
        """
        d = (adate-self.obs_date).days/365
        if d < self.pillars[0] or d > self.pillars[-1]:
            print (f"Cannot extrapolate rates (date: {adate}).")
            return None, None
        else:
            return d, self.interpolator(d)

    def forward_rate(self, d1, d2):
        """
        Compute the forward rate for the time period [d1, d2]
        
        Params:
        -------
        d1, d2: datetime.date
            start and end time of the period
        """
        d1, r1 = self.interp_rate(d1)
        d2, r2 = self.interp_rate(d2)
        if d1 is None or d2 is None:
            return None
        else:
            return (r2*d2 - r1*d1)/(d2 - d1)
    
class CreditCurve:
    """
    A class to represents credit curves

    Attributes:
    -----------
    obs_date: datetime.date
        observation date
    pillars: list(datetime.date)
        pillar dates of the curve
    ndps: list(float)
        non-default probabilities
    """    
    def __init__(self, obs_date, pillars, ndps):
        self.obs_date = obs_date
        if obs_date not in pillars:
            pillars = [obs_date] + pillars
            ndps = np.insert(ndps, 0, 1)
        self.pillars = [d.toordinal() for d in pillars]
        self.ndps = ndps
        self.interpolator = (self.pillars, self.ndps)
        
    def ndp(self, d):
        """
        Interpolates non-default probability at arbitrary dates

        Params:
        -------
        d: datatime.date
            the interpolation date
        """
        d_days = d.toordinal()
        if d < self.pillars[0] or d > self.pillar[-1]:
            print (f"Cannot extrapolate survival probabilities (date: {d}).")
            return None
        return self.interpolator(d_days)
    
    def hazard(self, d):
        """
        Computes the annualized hazard rate

        Params:
        -------
        d: datetime.date
            the date at which the hazard rate is computed
        """
        ndp_1 = self.ndp(d)
        ndp_2 = self.ndp(d + relativedelta(days=1))
        if ndp_1 is None or ndp_2 is None:
            return None
        delta_t = 1.0 / 365.0
        h = -1.0 / ndp_1 * (ndp_2 - ndp_1) / delta_t
        return h

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
    
