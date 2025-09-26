import numpy as np

from .global_const import GlobalConst

from dateutil.relativedelta import relativedelta
from scipy.interpolate import interp1d

class DiscountCurve:
    """
    A class to represent discount curves

    Attributes:
    -----------
    pillar_dates: list(datetime.date)
        pillars dates of the discount curve
    discount_factors: list(float)
        actual discount factors
    """
    def __init__(self, pillar_dates, discount_factors):
        discount_factors = np.array(discount_factors)
        if GlobalConst.OBSERVATION_DATE not in pillar_dates:
            pillar_dates = [GlobalConst.OBSERVATION_DATE] + pillar_dates
            discount_factors = np.insert(discount_factors, 0, 1)
        self.pillar_dates = pillar_dates
        self.pillars = [p.toordinal() for p in pillar_dates]
        self.dfs = discount_factors
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
            raise ValueError(f"Cannot extrapolate discount factors (date: {adate}).")
        return np.exp(self.interpolator(d))
    
    def rate(self, d):
        """
        Computes the annualized yield at a given date

        Params:
        -------
        d: datetime.date
            actual date at which calculate the yield
        """
        return -np.log(self.df(d))/((d-self.pillar_dates[0]).days/365) 
    
    def rates(self):
        """
        Computes the annualized yield corresponding to the provided discount factors
        """
        rs = []
        for i in range(1, len(self.dfs)):
            rs.append(-np.log(self.dfs[i])/((self.pillar_dates[i]-self.pillar_dates[0]).days/365))
        return rs

class TermStructure:
    """
    A class to represent a forward rate curve

    Attributes:
    -----------
    pillar_dates: list(datetime.date)
        pillar dates of the forward rate curve
    spot_rates: list(float)
        rates of the forward curve
    """
    def __init__(self, pillars, spot_rates):
        self.pillars_dates = pillars
        self.pillars = [(p-GlobalConst.OBSERVATION_DATE).days/365 for p in pillars]
        self.rates = spot_rates
        self.interpolator = interp1d(self.pillars, self.rates)
        
    def interp_rate(self, adate):
        """
        Find the rate at time d
        
        Params:
        -------
        adate : datetime.date
            date of the interpolated rate
        """
        d = (adate-GlobalConst.OBSERVATION_DATE).days/365
        if d < self.pillars[0] or d > self.pillars[-1]:
            raise ValueError(f"Cannot extrapolate rates (date: {adate}).")
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
        return (r2*d2 - r1*d1)/(d2 - d1)
            
class FlatTermStructure(TermStructure):
    """
    A class to represent a flat rate term structure

    Attributes:
    -----------
    end_date: list(datetime.date)
        end date of the flat structure
    flat_rate: list(float)
        rate of the forward curve
    """
    def __init__(self, end_date, rate):
        self.rate = rate
        super(FlatTermStructure, self).__init__([GlobalConst.OBSERVATION_DATE, end_date], [rate]*2)            

    def forward_rate(self, d1, d2):
        """
        Compute the forward rate for the time period [d1, d2]
        
        Params:
        -------
        d1, d2: datetime.date
            start and end time of the period
        """    
        dd1 = (d1-self.pillars_dates[0]).days/365
        if dd1 < self.pillars[0] or dd1 > self.pillars[-1]:
            raise ValueError(f"Cannot extrapolate rates (date: {d1}).")

        dd2 = (d2-self.pillars_dates[0]).days/365
        if dd2 < self.pillars[0] or dd2 > self.pillars[-1]:
            raise ValueError(f"Cannot extrapolate rates (date: {d2}).")
        return self.rate
        
class CreditCurve:
    """
    A class to represents credit curves

    Attributes:
    -----------
    pillars: list(datetime.date)
        pillar dates of the curve
    ndps: list(float)
        non-default probabilities
    """    
    def __init__(self, pillars, ndps):
        if GlobalConst.OBSERVATION_DATE not in pillars:
            pillars = [GlobalConst.OBSERVATION_DATE] + pillars
            ndps = np.insert(ndps, 0, 1)
        self.pillar_dates = pillars
        self.pillars = [d.toordinal() for d in pillars]
        self.ndps = ndps
        self.interpolator = interp1d(self.pillars, self.ndps)

    def ndp(self, d):
        """
        Interpolates non-default probability at arbitrary dates

        Params:
        -------
        d: datatime.date
            the interpolation date
        """
        d_days = d.toordinal()
        if d < GlobalConst.OBSERVATION_DATE or d_days > self.pillars[-1]:
            raise ValueError(f"Cannot extrapolate survival probabilities (date: {d}).")
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