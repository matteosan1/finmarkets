import numpy as np, pickle

from scipy.stats import norm, rv_continuous, binom, multivariate_normal
from scipy.integrate import quad
from scipy.interpolate import interp1d

from datetime import date
from dateutil.relativedelta import relativedelta

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

def maturity_from_str(maturity, unit="m"):
    """
    Utility to convert time intervals to integers in months. 
    The interval has the following format "XXy" with XX the value and y the units (y, Y, m, M, d, D).

    Params:
    -------
    maturity: str
        the string to be converted
    unit: str
        time unit of the output, default value is month
    """
    tag = maturity[-1].lower()
    maturity = float(maturity[:-1])
    if tag == "y":
        maturity *= 12
    elif tag == "d":
        maturity /= 30
    elif tag != "m":
        raise ValueError("Unrecognized label {}".format(tag))

    unit = unit.lower()
    if unit == "y":
        maturity /= 12
    elif unit == "d":
        maturity *= 30
    elif unit != "m":
        raise ValueError("Unrecognized output unit {}".format(unit))
    
    return maturity

def generate_dates(start_date, maturity, tenor="1y"):
    """
    Computes a set of dates given starting date and length in months.

    Params:
    -------
    start_date: datetime.date
        the start date of the set of dates
    maturity: str
        maturity that defines the length of the list of dates
    tenor: str
        tenor of the list of dates, by default is 12 months
    """
    maturity_months = int(round(maturity_from_str(maturity), 0))
    tenor_months = int(round(maturity_from_str(tenor), 0))
    dates = []
    for d in range(0, maturity_months, tenor_months):
        dates.append(start_date + relativedelta(months=d))
    dates.append(start_date + relativedelta(months=maturity_months))
    return dates

class Bond:
    """
    A class to represent bonds

    Params:
    -------
    start_date: datetime.date
        start date of the contract
    maturity: str
        maturity of the bond
    K: float or list(float)
        coupons of the bond
    tenor: str
        tenor of the coupon
    face_value: float
        face value of the bond, default value 100
    """
    def __init__(self, start_date, K, maturity, tenor, face_value=100, debug=False):
        self.start_date = start_date
        self.maturity = maturity
        self.payment_dates = generate_dates(start_date, maturity)
        self.face_value = face_value
        self.K = K
        self.tenor = tenor
        self.debug = debug

    def npv_K(self, K, dc, P):
        val = 0
        for i in range(1, len(self.payment_dates)):
            tau = (self.payment_dates[i] - self.payment_dates[i-1]).days/365
            val += K*tau*dc.df(self.payment_dates[i])
        val += dc.df(self.payment_dates[-1])
        val *= self.face_value
        return val - P

    def findK(self, dc, P):
        return brentq(self.npv_K, 0, 1, args=(dc, P))

    def npv(self, dc):
        """
        Computes the bond NPV
        
        Params:
        -------
        dc: DiscountCurve
            discount curve to be used in the pricing
        """
        val = 0
        for i in range(1, len(self.payment_dates)):
            tau = (self.payment_dates[i] - self.payment_dates[i-1]).days/365
            cpn = self.face_value*self.K*tau
            if i == len(self.payment_dates)-1:
                cpn += self.face_value
            if self.debug:
                print ("CPN Bond: ", i, cpn)
            val += cpn*dc.df(self.payment_dates[i])
        return val

    def npv_flat_default(self, dc, pd, R=0.4):
        """
        Computes the bond NPV in case of default
        
        Params:
        -------
        dc: DiscountCurve
            discount curve to be used in the pricing
        pd: float
            flat default probability of the issuer
        R: float
            recovery rate, default value 0.4
        """
        val = 0
        for i in range(1, len(self.payment_dates)):
            tau = (self.payment_dates[i] - self.payment_dates[i-1]).days/365
            cpn = self.face_value*self.K*tau
            if i == len(self.payment_dates)-1:
                cpn += self.face_value
            val += (pd*(1-pd)**(i-1)*R + (1-pd)**i)*cpn*dc.df(self.payment_dates[i])
            if self.debug:
                print ("CPN Bond: ", i, cpn)
        return val

    def loss(self, dc, def_date):
        """
        Computes the bond loss in case of default
        
        Params:
        -------
        dc: DiscountCurve
          discount curve to be used in the pricing
        def_date: datetime.date
          default date of the bond emitter
        """
        val = 0
        for i in range(1, len(self.payment_dates)):
            if self.payment_dates[i-1] <= def_date < self.payment_dates[i]:
                rateo = (self.payment_dates[i] - def_date).days/(self.payment_dates[i] - self.payment_dates[i-1]).days
                val += self.K*rateo*self.tau*dc.df(def_date)
            elif self.payment_dates[i] > def_date:  
                val += self.K*self.tau*dc.df(self.payment_dates[i])
        val += dc.df(self.payment_dates[-1])
        return self.FV*val
    
#def bond_value(N, C, r, maturity):
#    value = 0
#    for t in range(1, maturity+1):
#        value += N*C*1/(1+r)**t
#    value += N*1/(1+r)**t
#    return value

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
        self.obs_date = obs_date
        self.pillar_dates = [obs_date] + pillar_dates
        self.discount_factors = np.insert(np.array(discount_factors), 0, 1)
        self.log_discount_factors = [np.log(discount_factor) for discount_factor in self.discount_factors]
        self.pillar_days = [(pillar_date - obs_date).days for pillar_date in self.pillar_dates]
        
    def df(self, d):
        """
        Gets interpolated discount factor at `d`

        Params:
        -------
        d: datetime.date
            actual date at which we would like the interpolated discount factor
        """
        if d < self.obs_date or d > self.pillar_dates[-1]:
            print ("Cannot extrapolate discount factors (date: {}).".format(d))
            return None
        d_days = (d - self.obs_date).days
        interpolated_log_discount_factor = np.interp(d_days, self.pillar_days, self.log_discount_factors)
        return np.exp(interpolated_log_discount_factor)
    
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
        self.pillars = pillars
        self.pillar_days = [(p - obs_date).days/365 for p in pillars]
        self.rates = rates

    def interp_rate(self, d):
        """
        Find the rate at time d
        
        Params:
        -------
        d : datetime.date
            date of the interpolated rate
        """
        d_frac = (d - self.obs_date).days/365
        if d < self.obs_date or d_frac > self.pillar_days[-1]:
            print ("Cannot extrapolate rates (date: {}).".format(d))
            return None, None
        else:
            return d_frac, np.interp(d_frac, self.pillar_days, self.rates)

    def forward_rate(self, d1, d2):
        """
        Compute the forward rate for the time period [d1, d2]
        
        Params:
        -------
        d1, d2: datetime.date
            start and end time of the period
        """
        d1_frac, r1 = self.interp_rate(d1)
        d2_frac, r2 = self.interp_rate(d2)
        if d1_frac is None or d2_frac is None:
            return None
        else:
            return (r2*d2_frac - r1*d1_frac)/(d2_frac - d1_frac)
    
class OvernightIndexSwap:
    """
    A class to represent O/N swaps

    Attributes:
    -----------
    notional: float
        notional of the swap
    start_date: datetime.date
         start date of the contract
    maturity: str
        maturity of the swap.
    fixed_rate: float
        rate of the fixed leg of the swap
    """
    def __init__(self, nominal, start_date, maturity, fixed_rate):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.payment_dates = generate_dates(start_date, maturity)
      
    def npv_floating(self, dc):
        """
        Compute the floating leg NPV
    
        Params:
        -------
        dc: DiscountCurve
            discount curve to be used in the calculation
        """
        return self.nominal * (dc.df(self.payment_dates[0]) - dc.df(self.payment_dates[-1]))
  
    def npv_fixed(self, dc):
        """
        Computes the fixed leg NPV
    
        Params:
        -------
        dc: DiscountCurve
            discount curve to be used in the calculation
        """
        val = 0
        for i in range(1, len(self.payment_dates)):
            val += dc.df(self.payment_dates[i]) * \
                    (self.payment_dates[i] - self.payment_dates[i-1]).days/360 
        return self.nominal*self.fixed_rate*val
  
    def npv(self, dc):
        """
        Computes the contract NPV seen from the point of view of the 
        receiver of the floating leg.
    
        Params:
        -------
        dc: DiscountCurve
            discount curve to be used in the calculation
        """
        return self.npv_floating(dc) - self.npv_fixed(dc)

    def fair_value_strike(self, dc):
        """
        Computes the fair value strike

        Params:
        -------
        dc: DiscountCurve
            siscount curve object used for npv calculation.
        """
        den = self.npv_fixed_leg(dc)/self.fixed_rate
        num = self.npv_floating_leg(dc)
        return num/den
    
def call(St, K, r, sigma, ttm):
    """
    Compute call price through Black-Scholes formula

    Params:
    -------
    St: float
        underlying spot price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: str or list(str)
        time to maturity
    """
    if type(ttm) == list:
        ttm = np.array([maturity_from_str(t, "y") for t in ttm])
    else:
        ttm = maturity_from_str(ttm, "y")
    return (St*norm.cdf(d_plus(St, K, r, sigma, ttm)) -
            K*np.exp(-r*(ttm))*norm.cdf(d_minus(St, K, r, sigma, ttm)))

def put(St, K, r, sigma, ttm):
    """
    Computes put price through Black-Scholes formula

    Params:
    -------
    St: float
        underlying spot price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: str
        time to maturity
    """
    if type(ttm) == list:
        ttm = np.array([maturity_from_str(t, "y") for t in ttm])
    else:
        ttm = maturity_from_str(ttm, "y")
    return (K*np.exp(-r*(ttm))*norm.cdf(-d_minus(St, K, r, sigma, ttm)) -
            St*norm.cdf(-d_plus(St, K, r, sigma, ttm)))
    
def d_plus(St, K, r, sigma, ttm):
    """
    Computes d_plus coefficient for Black-Scholes formula

    Params:
    -------
    St: float
        underlying price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: float
        time to maturity in years
    """
    num = np.log(St/K) + (r + 0.5*sigma**2)*(ttm)
    den = sigma*np.sqrt(ttm)
    return num/den

def d_minus(St, K, r, sigma, ttm):
    """
    Computes d_minus coefficient for Black-Scholes formula

    Params:
    -------
    St: float
        underlying price
    K: float
        option strike
    r: float
        risk free interest rate
    sigma: float
        underlying volatility
    ttm: float
        time to maturity in years
    """
    return d_plus(St, K, r, sigma, ttm) - sigma*np.sqrt(ttm)

class ParAssetSwap:
    """
    A class to represent interest rate swaps

    Attributes:
    -----------
    bond_price: float
        market price of the underlying bond
    bond: Bond
        bond object underlying the asset swap
    tenor_float: str
        tenor of the float leg of the swap
    dc: DiscountCurve
        discount curve for pricing
    fc: ForwardRateCurve
        forward curve to value float leg
    """    
    def __init__(self, bond_price, bond, tenor_float, dc, fc, debug=False):
        self.bond = bond
        self.nominal = bond.face_value
        self.fixed_rate = bond.K
        self.fixed_dates = bond.payment_dates
        self.float_dates = generate_dates(bond.start_date, bond.maturity, tenor_float)
        self.dc = dc
        self.fc = fc
        self.debug = debug
        self.bond_price = bond_price
        self.asspread()
    
    def asspread(self):
        """
        Computes asset swap spread

        Params:
        -------
        """
        A = self.annuity()
        s = ((self.bond.npv(self.dc) - self.bond_price)/A)/self.bond.face_value
        self.spread = s

    def annuity(self):
        """
        Computes the annuity

        Params:
        -------
        """
        a = 0
        for i in range(1, len(self.fixed_dates)):
            tau = (self.fixed_dates[i] - self.fixed_dates[i-1]).days / 360
            a += self.dc.df(self.fixed_dates[i])*tau
        return a

    def swap_rate(self):
        """ 
        Computes swap rate

        Params:
        -------
        """
        A = self.annuity()
        num = 0
        for j in range(1, len(self.float_dates)):
            F = self.fc.forward_rate(self.float_dates[j], self.float_dates[j-1])
            tau = (self.float_dates[j] - self.float_dates[j-1]).days / 360
            D = self.dc.df(self.float_dates[j])
            num += (F+self.spread) * tau * D
        return num/A

    def npv(self):
        """
        Computes the swap NPV
        
        Params:
        -------
        """
        S = self.swap_rate()
        A = self.annuity()
        return (self.bond_price-self.bond.face_value) + self.nominal * (S - self.fixed_rate) * A

    def float_flows(self):
        val = 0
        for i in range(1, len(self.float_dates)):
            F = self.fc.forward_rate(self.float_dates[i], self.float_dates[i-1])
            tau = (self.float_dates[i] - self.float_dates[i-1]).days / 360
            cpn = (F+self.spread)*self.nominal*tau
            if self.debug:
                print ("Swap Fixed: ", self.float_dates[i], cpn)
            val += cpn*dc.df(self.floating_leg_dates[i])
        if self.debug:
            print ("Total float", val)
        return val

    def fixed_flows(self):
        val = 0
        for i in range(1, len(self.fixed_dates)):
            tau = (self.fixed_dates[i] - self.fixed_dates[i-1]).days / 360
            cpn = self.fixed_rate*self.nominal*tau
            if self.debug:
                print ("Swap Fixed: ", i, cpn)
            val += cpn*dc.df(self.fixed_dates[i])
        if self.debug:
            print ("Total fixed ", val)
        return val
    
class InterestRateSwap:
    """
    A class to represent interest rate swaps

    Attributes:
    -----------
    nominal: float
        nominal of the swap
    start_date: datetime.date
        starting date of the contract
    maturity: str
        maturity of the swap.
    fixed_rate: float
        rate of the fixed leg of the swap
    tenor_float: str
        tenor of the float leg
    tenor_fix: str
        tenor of the fixed leg. default value is 1 year
    """    
    def __init__(self, nominal, start_date, maturity,
                 fixed_rate, tenor_float, tenor_fix="12m"):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.fixed_leg_dates = generate_dates(start_date, maturity, tenor_fix)
        self.floating_leg_dates = generate_dates(start_date, maturity, tenor_float)

    def annuity(self, dc):
        """
        Computes the fixed leg annuity

        Params:
        -------
        dc: DiscountCurve
            discount curve object used for the annuity
        """
        a = 0
        for i in range(1, len(self.fixed_leg_dates)):
            a += dc.df(self.fixed_leg_dates[i])
        return a

    def npv(self, dc, fc):
        """
        Computes the NPV of the swap

        Params:
        -------
        dc: DiscountCurve
            discount curve to be used in the calculation
        fc: ForwardRateCurve
            forward curve
        """
        S = self.swap_rate(dc, fc)
        A = self.annuity(dc)
        return self.nominal * (S - self.fixed_rate) * A

    def swap_rate(self, dc, fc):
        """
        Compute the swap rate of the IRS

        Params:
        -------
        dc: DiscountCurve
            discount curve object used for swap rate calculation
        fc: ForwardRateCurve
            forward curve object used for swap rate calculation
        """
        num = 0
        for j in range(1, len(self.floating_leg_dates)):
            F = fc.forward_rate(self.floating_leg_dates[j], self.floating_leg_dates[j-1])
            tau = (self.floating_leg_dates[j] - self.floating_leg_dates[j-1]).days / 360
            D = dc.df(self.floating_leg_dates[j])
            num += F * tau * D
        return num/self.annuity(dc)

class InterestRateSwaption:
    """
    A class to represent interest rate swaptions

    Attributes:
    -----------
    nominal: float
        nominal of the swap
    start_date: datetime.date
        start date of contract
    exercise_date: datetime.date
        exercise date of the swaptions
    maturity: str
        maturity of the swap
    volatility: float
        swap rate volatility
    fixed_rate: float
        rate of the fixed leg of the swap
    tenor: str
        tenor of the contract
    """
    def __init__(self, nominal, start_date, exercise_date, maturity,
                 volatility, fixed_rate, tenor):
        self.irs = InterestRateSwap(nominal, start_date, maturity, fixed_rate, tenor)
        self.exercise_date = exercise_date
        self.sigma = volatility
        
    def payoffBS(self, obs_date, dc, fc):
        """
        Estimates the swaption NPV using Black-Scholes formula
        
        Params:
        -------
        obs_date: datetime.date
            observation date
        dc: DiscountCurve
            curve to discount the npv
        fc: ForwardRateCurve
            forward curve to compute the swap rate
        """
        T = (self.exercise_date - obs_date).days/365
        N = self.irs.nominal
        K = self.irs.fixed_rate
        S = self.irs.swap_rate(dc, fc)
        A = self.irs.annuity(dc)
        dp = (np.log(S/K) + 0.5*self.sigma**2*T)/(self.sigma*np.sqrt(T))
        dm = (np.log(S/K) - 0.5*self.sigma**2*T)/(self.sigma*np.sqrt(T))
        return N*A*(S*norm.cdf(dp)-K*norm.cdf(dm))
    
    def payoffMC(self, obs_date, dc, fc, n_scenarios=10000):
        """
        Estimates the swaption NPV with Monte Carlo Simulation
        
        Params:
        -------
        obs_date: datetime.date
            observation date
        dc: DiscountCurve
            the curve to discount the npv
        fc: ForwardRateCurve
            forward curve to compute the swap rate
        n_scenarios: int
            number of Monte Carlo experiment to simulate
        """
        T = (self.exercise_date - obs_date).days/365
        payoffs = []
        S0 = self.irs.swap_rate(dc, fc)
        for _ in range(n_scenarios):
            S = S0 * np.exp(-self.sigma**2/2*T + self.sigma*np.random.normal()*np.sqrt(T))
            payoff = self.irs.nominal*max(0, S - self.irs.fixed_rate)*self.irs.annuity(dc)
            payoffs.append(payoff)
        payoff = np.mean(payoffs)
        one_sigma = np.std(payoffs)/np.sqrt(n_scenarios)
        return payoff, one_sigma

class CreditCurve:
    """
    A class to represents credit curves

    Attributes:
    -----------
    obs_date: datetime.date
        observation date
    pillar_date: list(datetime.date)
        pillar dates of the curve
    ndps: list(float)
        non-default probabilities
    """    
    def __init__(self, obs_date, pillar_dates, ndps):
        self.obs_date = obs_date
        self.pillar_dates = [obs_date] + pillar_dates
        self.pillar_days = [(pd - obs_date).days for pd in self.pillar_dates]
        self.ndps = np.insert(np.array(ndps), 0, 1)
        
    def ndp(self, d):
        """
        Interpolates non-default probability at arbitrary dates

        Params:
        -------
        d: datatime.date
            the interpolation date
        """
        d_days = (d - self.obs_date).days
        if d < self.obs_date or d_days > self.pillar_days[-1]:
            print ("Cannot extrapolate survival probabilities (date: {}).".format(d))
            return None
        return np.interp(d_days, self.pillar_days, self.ndps)
    
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
        delta_t = 1.0 / 365.0
        h = -1.0 / ndp_1 * (ndp_2 - ndp_1) / delta_t
        return h

class CreditDefaultSwap:
    """
    A class to represent Credit Default Swaps

    Attributes:
    -----------
    nominal: float
        nominal of the swap
    start_date: datetime.date
        starting date of the contract
    maturity: str
        maturity of the swap.
    fixed_spread: float
        the spread associated to the premium leg
    tenor: str
        tenor of the premium leg, default is 3m
    recovery: float
        recovery parameter in case of default, default value is 40%
    """    
    def __init__(self, nominal, start_date, maturity, fixed_spread,
                 tenor="3m", recovery=0.4):
        self.nominal = nominal
        self.payment_dates = generate_dates(start_date, maturity, tenor)
        self.fixed_spread = fixed_spread
        self.recovery = recovery

    def npv_premium_leg(self, dc, cc):
        """
        Valuate the premium leg

        Params:
        -------
        dc: DiscountCurve 
            the curve to discount the NPV
        cc: CreditCurve
            the curve to extract the default probabilities
        """
        npv = 0
        for i in range(1, len(self.payment_dates)):
            npv += (dc.df(self.payment_dates[i]) *
                    cc.ndp(self.payment_dates[i]))
        return self.fixed_spread * npv * self.nominal

    def npv_default_leg(self, dc, cc):
        """
        Valuate the default leg

        Params:
        -------
        dc: DiscountCurve 
            the curve to discount the NPV
        cc: CreditCurve
            the curve to extract the default probabilities
        """
        npv = 0
        d = self.payment_dates[0]
        while d < self.payment_dates[-1]:
            npv += dc.df(d) * (
                   cc.ndp(d) -
                   cc.ndp(d + relativedelta(days=1)))
            d += relativedelta(days=1)
        return npv * self.nominal * (1 - self.recovery)

    def npv(self, dc, cc):
        """
        Valuate the CDS

        Params:
        -------
        dc: DiscountCurve 
            the curve to discount the NPV
        cc: CreditCurve
            the curve to extract the default probabilities
        """
        return self.npv_default_leg(dc, cc) - self.npv_premium_leg(dc, cc)

    def breakevenRate(self, dc, cc):
        """
        Compute the swap breakeven

        Params:
        -------
        dc: DiscountCurve 
            the curve to discount the NPV
        cc: CreditCurve
            the curve to extract the default probabilities
        """
        num = self.npv_default_leg(dc, cc)
        den = self.npv_premium_leg(dc, cc)/self.fixed_spread
        return num/den

class ExpDefault(rv_continuous):
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

def gaussian_copula_default(N, Q, pillars, cov, n_def, simulations=10000, seed=1):
    """
    Utility function to estimate nth-to-default survival probability in 
    presence of Gaussian correlation using MC simulation.
    
    Params:
    -------
    N: int
        number of entities
    Q: scipy.stats.rv_continuous
        marginal cumulative default probability
    pillars: list(datetime.date)
        list of dates at which survival probality is computed
    cov: numpy.array
        covariance matrix for Gaussian correlation
    n_def: int
        number of defaults
    simulations: int
        number of simulations used to compute the probability, default is 10000
    seed: int
        seed for random numbers, default value is 1
    """
    np.random.seed(seed)

    if n_def == 0 or N == 0:
        raise ValueError("Number of defaults or entities cannot be zero !")

    if N < n_def:
        n_def = N
        print ("Warning: default number set equal to number of entities.")

    mv = multivariate_normal(mean=np.zeros(N), cov=cov)
    x = mv.rvs(size = simulations)
    x_unif = norm.cdf(x)

    default_times = np.sort(Q.ppf(x_unif))
    Ts = [(p - pillars[0]).days/365 for p in pillars]
    S_corr = np.array([(1-(default_times[:, n_def-1] <= t).mean()) for t in Ts])
    return S_corr
    
class BasketDefaultSwaps:
    """
    A class to valuate basket default swaps

    Attributes:
    -----------
    nominal: float
        nominal of the swap
    N: int
        number of reference entities underlying the BDS
    hazard_rate: float
        annualized hazard rate
    rho: float
        correlation factor for the defaults
    start_date: datetime.date
        starting date of the contract
    maturity: str
        maturity of the swap
    spread: float
        spread associated to the premium leg
    tenor: str
        tenor of the premium leg, default is 3m
    recovery: float
        recovery parameter in case of default, default value is 40%
    """    
    def __init__(self, nominal, N, hazard_rate, rho, start_date, maturity,
                 spread, tenor="3m", recovery=0.4):
        self.cds = CreditDefaultSwap(nominal, start_date, maturity,
                                     spread, tenor, recovery)
        self.Q = ExpDefault(l=hazard_rate)
        self.N = N
        self.rho = rho
        self.cc = None

    def credit_curve(self, obs_date, pillars, n_defaults):
        """
        Computes the credit curve needed for the BDS valuation

        Params:
        -------
        obs_date: datetime.date
            observation date
        pillars: list(datetime.date) 
            pillar dates to determine credit curve
        n_defaults: int
            number of defaults required by the BDS
        """
        simulations = 100000
        cov = np.ones(shape=(self.N, self.N))*self.rho
        np.fill_diagonal(cov, 1)
        mean = np.zeros(self.N)
        mv = multivariate_normal(mean=mean, cov=cov)
        x = mv.rvs(size=simulations)
        x_unif = norm.cdf(x)
        default_times = self.Q.ppf(x_unif)

        Ts = [(p-obs_date).days/360 for p in pillars]
        ndps = []
        for t in Ts:
            b = np.count_nonzero(default_times<=t, axis=1)
            ndps.append(1 - len(b[b>=n_defaults])/simulations)
        self.cc = CreditCurve(obs_date, pillars, ndps)

    def npv(self, dc):
        """
        Computes the npv of the BDS

        Params:
        -------
        dc: DiscountCurve 
            discount curve to valuate the contract
        """
        if self.cc is None:
            print ("Need to call credit_curve method first !")
            return None
        return self.cds.npv(dc, self.cc)
  
    def breakeven(self, dc):
        """
        Computes the breakeven of the BDS

        Params:
        -------
        dc: DiscountCurve 
            discount curve to valuate the contract
        """
        return self.cds.breakevenRate(dc, self.cc)
    
class BasketDefaultSwapsOneFactor:
    """
    A class to represent basket default swaps whose valuation relies on the Gaussian One Factor Model

    Attributes:
    -----------
    nominal: float
        notional of the swap
    N: int
        reference entities underlying the BDS
    rho: float
        correlation factor for the defaults
    start_date: datetime.date
        starting date of the contract
    spread: float
        spread associated to the premium leg
    maturity: str
        maturity of the swap
    tenor: int
        tenor of the premium leg in months, default is 3
    recovery: float
        recovery parameter in case of default, default value is 40%
    """
    def __init__(self, nominal, N, rho, start_date, maturity,
                 spread, tenor="3m", recovery=0.4):
        self.N = N
        self.rho = rho
        self.cds = CreditDefaultSwap(nominal, start_date, maturity,
                                     spread, tenor, recovery)

    def one_factor_model(self, M, f, obs_date,
                         Q_dates, Q, dc, ndefaults):
        """
        Estimates a quantity value according to the One Factor Model

        Params:
        ------- 
        M: int
            market value
        f: user defined function
            function returning the quantity to estimate
        obs_date: datetime.date
            observation da
        Q_dates: list(datetime.date)
            list of dates with known default probability
        Q: list(float)
            default probabilities
        dc: DiscountCurve
            discount curve
        ndefaults: int
            number of "required" default in the basket definition
        """
        P = norm.cdf((norm.ppf(Q) - np.sqrt(self.rho)*M)/np.sqrt(1-self.rho))
        b = binom(self.N, P)
        S = 1 - (1 - b.cdf(ndefaults-1))
        cc = CreditCurve(obs_date, Q_dates, S)
        return f(dc, cc)*norm.pdf(M)
            
    def breakeven(self, obs_date, Q_dates, Q, dc, ndefaults):
        """
        Computes the breakeven of the BDS

        Params:
        -------
        obs_date: datetime.date
            observation date
        Q_dates: list(datetime.date)
            list of dates with known default probability
        Q: list(float)
            default probabilities
        dc: DiscountCurve
            discount curve
        ndefaults: int
            number of "required" default in the basket definition
        """        
        s = quad(self.one_factor_model, -np.inf, np.inf, 
                 args=(self.cds.breakevenRate, Q_dates, Q, dc, ndefaults))
        return s[0]
    
    def npv(self, obs_date, Q_dates, Q, dc, ndefaults):
        """
        Computes the npv of the BDS

        Params:
        -------
        obs_date: datetime.date
            observation date
        Q_dates: list(datetime.date)
            list of dates with known default probability
        Q: list(float)
            default probabilities
        dc: DiscountCurve
            discount curve
        ndefaults: int
            number of "required" default in the basket definition
        """
        s = quad(self.one_factor_model, -np.inf, np.inf, 
                 args=(self.cds.npv, Q_dates, Q, dc, ndefaults))
        return s[0]        

class CollDebtObligation:
    """
    A class to handle synthetic CDOs, whose valuation relies on One Factor Model

    Params:
    -------
    nominal: float
        nominal of the contract
    N: int
        number of underlying enties
    tranches; list(tuple)
        tranches definition in terms of percentage of the nominal
    rho: float
        correlation in names default probabilities
    cc:: CreditCurve
        credit curve of the underlyings
    start:date: datetime.date
        start date of the contract
    spreads: list(floats)
        list of spreads for each tranche
    maturity: str
        maturity of the contract
    tenor: str
        tenor of the contract, default value is 3 months
    recovery: float
        percentage of the contract value that will be recovered in case of default, default value 0.4
    """
    def __init__(self, nominal, N, tranches, rho, cc,
                 start_date, spreads, maturity, tenor="3m", recovery=0.4):
        self.nominal = nominal
        self.N = N
        self.tranches = tranches
        self.payment_dates = generate_dates(start_date, maturity, tenor)
        self.spreads = spreads
        self.rho = rho
        self.recovery = recovery
        self.cc = cc

    def one_factor_model(self, M, Q, l, L, U):
        """
        Implements the One Factor Model.

        Params:
        -------
        M: float
            market condition parameter
        Q: list(float)
            default probabilities list
        l: int
            number of defaults
        L: float
            lower bound of the tranche
        U: float
            upper bound of the tranche
        """
        P = norm.cdf((norm.ppf(Q) - np.sqrt(self.rho) * M) / (np.sqrt(1 - self.rho)))
        b = binom(self.N, P)
        return b.pmf(l) * norm.pdf(M) * max(min(l/self.N *
               self.nominal * (1 - self.recovery), U) - L, 0)

    def expected_tranche_loss(self, d, L, U):
        """
        Computes the expected tranche loss

        Params:
        -------
        d: datetime.date
            valuation date
         L: float
            lower bound of the tranche
        U: float
            upper bound of the tranche
        """       
        Q = 1 - self.cc.ndp(d)
        v = 0 
        for l in range(self.N+1):
            i = quad(self.one_factor_model, -np.inf, np.inf,
                args=(Q, l, L, U))[0]
            v += i
        return v

    def npv_premium(self, tranche, dc):
        """
        Computes the NPV of the premium leg
        
        Params:
        -------
        tranche: int
            index of the tranche
        dc: DiscountCurve
            discount curve to be used in the calculation
        """
        L = self.tranches[tranche][0] * self.nominal
        U = self.tranches[tranche][1] * self.nominal
        v = 0
        for i in range(1, len(self.payment_dates)):
            ds = self.payment_dates[i - 1]
            de = self.payment_dates[i]
            D = dc.df(de)
            ETL = self.expected_tranche_loss(ds, L, U)
            v += D * (de - ds).days / 360 * max((U - L) - ETL, 0)
        return v * self.spreads[tranche]

    def npv_default(self, tranche, dc):
        """
        Computes the NPV of the default leg
        
        Params:
        -------
        tranche: int
            index of the tranche
        dc: DiscountCurve
            discount curve to be used in the calculation
        """

        U = self.tranches[tranche][1] * self.nominal
        L = self.tranches[tranche][0] * self.nominal
        v = 0
        for i in range(1, len(self.payment_dates)):
            ds = self.payment_dates[i - 1]
            de = self.payment_dates[i]
            ETL1 = self.expected_tranche_loss(ds, L, U)
            ETL2 = self.expected_tranche_loss(de, L, U)
            v += dc.df(de) * (ETL2 - ETL1)
        return v

    def npv(self, tranche, dc):
        """
        Computes the NPV of the contract
        
        Params:
        -------
        tranche: int
            index of the tranche
        dc: DiscountCurve
            discount curve to be used in the calculation
        """        
        return self.npv_default(tranche, dc) - self.npv_premium(tranche, dc)

    def fair_value(self, tranche, dc):
        """
        Computes the fair value of the contract
        
        Params:
        -------
        tranche: int
            index of the tranche
        dc: DiscountCurve
            discount curve to be used in the calculation
        """        
        num = self.npv_default(tranche, dc)
        den = self.npv_premium(tranche, dc) / self.spreads[tranche]
        return num / den

def var_continuous(f, alpha=0.95):
    """
    Computes VaR at a specified confidence level, given a continuous loss distribution
    
    Params:
    -------
    f: scipy.stats.rv_continuous
        continuous distribution representing the portfolio losses
    alpha: float
        confidence level for VaR calculation
    """
    return -f.ppf(1-alpha)

def es_continuous(f, alpha=0.95):
    """
    Computes Expected Shortfall at a specified confidence level, given a continuous loss distribution
    
    Params:
    -------
    f: scipy.stats.rv_continuous
        continuous distribution representing the portfolio losses
    alpha: float
        confidence level for ES calculation
    """
    def integrand(x, f):
        return f.ppf(x)
    alpha = 1-alpha
    I = quad(integrand, 0, alpha, args=(f,))
    return -1/alpha*I[0]
  
def generate_returns(df, N=10000):
    data = df.reset_index()
    return data.loc[np.random.choice(range(len(data)), N)]

def var_discrete(df, alpha=0.95, return_col="P", N=10000):
    """
    Computes VaR at a specified confidence level, given a discrete loss distribution as a DataFrame
    
    Params:
    -------
    df: pandas.DataFrame
        dataset containing the loss descrete distribution
    alpha: float
        confidence level for VaR calculation
    return_col: str
        name of the column in the dataframe
    N: int
        number of samples to generate
    """
    alpha = 1-alpha
    new_df = generate_returns(df, N)
    print (new_df.head())
    return -np.percentile(new_df[return_col], alpha*100)
    
def es_discrete(df, alpha=0.95, return_col="P", N=10000):
    """
    Computes Expected Shortfall at a specified confidence level, given a discrete loss distribution
    
    Params:
    -------
    df: pandas.DataFrame
        dataset containing the loss descrete distribution
    alpha: float
        confidence level for ES calculation
    return_col: str
        name of the column in the dataframe
    N: int
        number of samples to generate
    """
    alpha = 1-alpha
    new_df = generate_returns(df, N)
    var = np.percentile(new_df[return_col], alpha*100)	
    return -df[df<=var][return_col].mean()
