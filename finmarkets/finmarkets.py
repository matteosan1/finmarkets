import numpy as np

from scipy.stats import norm, rv_continuous, binom
from scipy.integrate import quad
from scipy.interpolate import interp1d

from datetime import date
from dateutil.relativedelta import relativedelta

def maturity_from_str(maturity_str):
    tag = maturity_str[-1].lower()
    maturity = int(maturity[:-1])
    if tag == "y":
        return maturity*12
    elif tag == "m":
        return maturity
    elif tag == "d":
        return maturity//30
    else:
        raise ValueError("Unrecognized label {}".format(tag))

class Bond:
    """
    Bond - class to price bonds.

    Params:
    -------
    face_value: float
        Face value of the bond.
    start_date: datetime.date
        Start date of the contract.
    maturity: str
        Maturity of the bond.
    coupon: float or list(float)
        Coupons of the bond.
    tenor: str
        Tenor of the coupon.
    """
    def __init__(self, face_value, start_date, maturity, coupon, tenor):
        self.FV = face_value
        self.C = coupon
        self.tau = maturity_from_str(tenor)/12
        self.payment_dates = generate_dates(start_date, maturity, tenor)

    def npv(self, dc):
        """
        npv - computes the bond NPV.
        
        Params:
        -------
        dc: DiscountCurve
          Discount curve to be used in the pricing.
        """
        val = 0
        for i in range(1, len(self.payment_dates)):
            val += self.C*self.tau*dc.df(self.payment_dates[i])
        val += dc.df(self.payment_dates[-1])
        return self.FV*val

    def loss(self, dc, def_date):
        """
        npv - computes the bond NPV.
        
        Params:
        -------
        dc: DiscountCurve
          Discount curve to be used in the pricing.
        def_date: datetime.date
          Simulated default date of the bond.
        """
        val = 0
        for i in range(1, len(self.payment_dates)):
            if self.payment_dates[i-1] <= def_date < self.payment_dates[i]:
                rateo = (self.payment_dates[i] - def_date).days/(self.payment_dates[i] - self.payment_dates[i-1]).days
                val += self.C*rateo*self.tau*dc.df(def_date)
            elif self.payment_dates[i] > def_date:  
                val += self.C*self.tau*dc.df(self.payment_dates[i])
        val += dc.df(self.payment_dates[-1])
        return self.FV*val
                
#def bond_value(N, C, r, maturity):
#    value = 0
#    for t in range(1, maturity+1):
#        value += N*C*1/(1+r)**t
#    value += N*1/(1+r)**t
#    return value

def generate_dates(start_date, maturity, tenor="1y"):
    """
    generate_swap_dates: computes a set of dates given starting date and length in months.
                         The tenor is by construction 12 months.

    Params:
    -------
    start_date: datetime.date
        The start date of the set of dates.
    maturity: str
        Maturity that define the length of the list of dates.
    tenor: str
        Tenor of the list of dates, by default it is 12 months.
    """
    maturity_monts = maturity_from_str(maturity)
    tenor_months = maturity_from_str(tenor)
    dates = []
    for d in range(0, maturity_months, tenor_months):
        dates.append(start_date + relativedelta(months=d))
    dates.append(start_date + relativedelta(months=maturity_months))
    return dates

class DiscountCurve:
    """
    DiscountCurve: class that manage discount curves.

    Attributes:
    -----------
    obs_date: datetime.date
        Observation date.
    pillar_dates: list(datetime.date)
        List of pillars of the discount curve.
    discount_factors: list(float)
        List of the actual discount factors.
    """
    def __init__(self, obs_date, pillar_dates, discount_factors):
        self.obs_date = obs_date
        self.pillar_dates = [obs_date] + pillar_dates
        self.discount_factors = [1.] + discount_factors
        self.log_discount_factors = [np.log(discount_factor) for discount_factor in self.discount_factors]
        self.pillar_days = [(pillar_date - obs_date).days for pillar_date in self.pillar_dates]
        
    def df(self, d):
        """
        df: method to get interpolated discount factor at `d`.

        Params:
        -------
        d: datetime.date
            The actual date at which we would like the interpolated discount factor.
        """
        d_days = (d - self.obs_date).days
        if d_days < self.obs_date or d_days > self.pillar_days[-1]:
            print ("Cannot extrapolate discount factors.")
            return None
        interpolated_log_discount_factor = np.interp(d_days, self.pillar_days, self.log_discount_factors)
        return np.exp(interpolated_log_discount_factor)
    
    def annualized_yield(self, d):
        """
        annualized_yield: computes the annualized yield at a given date.

        Params:
        -------
        d: datetime.date
        """
        return -np.log(self.df(d))/((d-self.obs_date).days/365) 

def makeDCFromDataFrame(df, obs_date, pillar_col='months', df_col='dfs'):
    """
    makeDCFromDataFrame - utility to create a DiscountCurve object from a pandas.DataFrame.
    
    Params:
    -------
    df: pandas.DataFrame
        Input pandas.DataFrame.
    obs_date: datetime.date
        Observation date.
    pillar_col: str
        Name of the pillar column in df, default is 'months'.
    df_col: str
        Name of discount factors column in df, default is 'dfs'.
    """
    pillars = [today + relativedelta(months=i) for i in df[pillar_col]]
    dfs = df[df_col]
    return DiscountCurve(obs_date, pillars, dfs)

class ForwardRateCurve:
    """
    ForwardRateCurve: container for a forward rate curve.

    Attributes:
    -----------
    obs_date: datetime.date
        Observation date.
    pillar_dates: list of datetime.date
        List of pillars of the forward rate curve.
    pillar_rates: list of rates
        List of rates of the forward curve.
    """
    def __init__(self, obs_date, pillars, rates):
        self.obs_date = obs_date
        self.pillars = pillars
        self.pillar_days = [(p - obs_date).days/365 for p in pillars]
        self.rates = rates

    def interp_rate(self, d):
        """
        interp_rate: find the rate at time d.
        
        Params:
        -------
        d : datetime.date
            The date of the interpolated rate.
        """
        d_frac = (d - self.obs_date).days/365
        if d_frac < self.obs_date or d_frac > self.pillar_days[-1]:
            print ("Cannot extrapolate rates.")
            return None, None
        else:
            return d_frac, np.interp(d_frac, self.pillar_days, self.rates)

    def forward_rate(self, d1, d2):
        """
        forward_rate: compute the forward rate for the time period [d1, d2].
        
        Params:
        -------
        d1, d2: datetime.date
            The start and end time of the period.
        """
        d1_frac, r1 = self.interp_rate(d1)
        d2_frac, r2 = self.interp_rate(d2)
        if d1_frac is None or d2_frac is None:
            return None
        else:
            return (r2*d2_frac - r1*d1_frac)/(d2_frac - d1_frac)
    
class OvernightIndexSwap:
    """
    OvernightIndexSwap: a class to valuate Overnight Index Swaps

    Attributes:
    -----------
    notional: float
        Notional of the swap.
    start_date: datetime.date
         Start date of the swap
    maturity: str
        Maturity of the swap.
    fixed_rate: float
        Rate of the fixed leg of the swap.
    """
    def __init__(self, nominal, start_date, maturity, fixed_rate):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.payment_dates = generate_dates(start_date, maturity)
      
    def npv_floating(self, dc):
        """
        Method to compute the floating leg NPV
    
        Parameters:
        dc: DiscountCurve
            Discount curve to be used in the calculation
        """
        return self.nominal * (dc.df(self.payment_dates[0]) - dc.df(self.payment_dates[-1]))
  
    def npv_fixed(self, dc):
        """
        Method to compute the fixed leg NPV
    
        Parameters:
        dc: DiscountCurve
            Discount curve to be used in the calculation
        """
        val = 0
        for i in range(1, len(self.payment_dates)):
            val += dc.df(self.payment_dates[i]) * \
                    (self.payment_dates[i] - self.payment_dates[i-1]).days/360 
        return self.nominal*self.fixed_rate*val
  
    def npv(self, dc):
        """
        Method to compute the contract NPV seen from the point of view of the 
        receiver of the floating leg.
    
        Parameters:
        dc: DiscountCurve
            Discount curve to be used in the calculation
        """
        return self.npv_floating(dc) - self.npv_fixed(dc)

    def fair_value_strike(self, dc):
        """
        Method to compute the fair value strike.

        Params:
        -------
        dc: DiscountCurve
            Discount curve object used for npv calculation.
        """
        den = self.npv_fixed_leg(dc)/self.fixed_rate
        num = self.npv_floating_leg(dc)
        return num/den
    
def call(St, K, r, sigma, ttm):
    """
    call: compute call price through Black-Scholes formula

    Params:
    -------
    St: float
        Underlying price.
    K: float
        Option strike.
    r: float
        Interest rate.
    sigma: float
        Underlying volatility.
    ttm: float
        Time to maturity in years.
    """
    return (St*norm.cdf(d_plus(St, K, r, sigma, ttm)) -
            K*np.exp(-r*(ttm))*norm.cdf(d_minus(St, K, r, sigma, ttm)))

def put(St, K, r, sigma, ttm):
    """
    put: compute put price through Black-Scholes formula

    Params:
    -------
    St: float
        Underlying price.
    K: float
        Option strike.
    r: float
        Interest rate.
    sigma: float
        Underlying volatility.
    ttm: float
        Time to maturity in years.
    """
    return (K*np.exp(-r*(ttm))*norm.cdf(-d_minus(St, K, r, sigma, ttm)) -
            St*norm.cdf(-d_plus(St, K, r, sigma, ttm)))
    
def d_plus(St, K, r, sigma, ttm):
    """
    d_plus: compute d_plus coefficient for Black-Scholes formula

    Params:
    -------
    St: float
        Underlying price.
    K: float
        Option strike.
    r: float
        Interest rate.
    sigma: float
        Underlying volatility.
    ttm: float
        Time to maturity in years.
    """
    num = np.log(St/K) + (r + 0.5*sigma**2)*(ttm)
    den = sigma*np.sqrt(ttm)
    return num/den

def d_minus(St, K, r, sigma, ttm):
    """
    d_minus: compute d_minus coefficient for Black-Scholes formula

    Params:
    -------
    St: float
        Underlying price.
    K: float
        Option strike.
    r: float
        Interest rate.
    sigma: float
        Underlying volatility.
    ttm: float
        Time to maturity in years.
    """
    return d_plus(St, K, r, sigma, ttm) - sigma*np.sqrt(ttm)
    
class InterestRateSwap:
    """
    InterestRateSwap: a class to valuate Interest Rate Swaps

    Attributes:
    -----------
    nominal: float
        Nominal of the swap.
    start_date: datetime.date
        Starting date of the contract.
    maturity: str
        Maturity of the swap.
    fixed_rate: float
        Rate of the fixed leg of the swap.
    tenor: str
        Tenor of the contract.
    """    
    def __init__(self, nominal, start_date, maturity, fixed_rate, tenor):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.fixed_leg_dates = generate_dates(start_date, maturity)
        self.floating_leg_dates = generate_dates(start_date, maturity, tenor)

    def annuity(self, dc):
        """
        annuity: compute the annuity.

        Params:
        -------
        dc: DiscountCurve
            Discount curve object used for the annuity.
        """
        a = 0
        for i in range(1, len(self.fixed_leg_dates)):
            a += dc.df(self.fixed_leg_dates[i])
        return a

    def npv(self, dc, fc):
        S = self.swap_rate(dc, fc)
        A = self.annuity(dc)
        return self.nominal * (S - self.fixed_rate) * A

    def swap_rate(self, discount_curve, euribor_curve):
        """
        swap_rate: compute the swap rate of the IRS.

        Params:
        -------
        dc: DiscountCurve
            Discount curve object used for swap rate calculation.
        fc: ForwardRateCurve
            Ibor curve object used for swap rate calculation.
        """
        num = 0
        for j in range(1, len(self.floating_leg_dates)):
            F = fc.forward_rate(self.floating_leg_dates[j], self.floating_leg_dates[j-1])
            tau = (self.floating_leg_dates[j] - self.floating_leg_dates[j-1]).days / 360
            D = dc.df(self.floating_leg_dates[j])
            num += F * tau * D
        return self.num(dc, fc) / self.annuity(dc)

class InterestRateSwaption:
    """
    InterestRateSwaption: class to manage swaptions.

    Attributes:
    -----------
    nominal: float
        Nominal of the swap.
    start_date: datetime.date
        The start date of contract.
    exercise_date: datetime.date
        The exercise date of the swaptions.
    maturity: str
        Maturity of the swap.
    volatility: float
        The swap rate volatility.
    fixed_rate: float
        Rate of the fixed leg of the swap.
    tenor: str
        Tenor of the contract.
    """
    def __init__(self, nominal, start_date, exercise_date, maturity,
                 volatility, fixed_rate, tenor):
        self.irs = InterestRateSwap(nominal, start_date, maturity, fixed_rate, tenor)
        self.exercise_date = exercise_date
        self.sigma = volatility
        
    def payoffBS(self, obs_date, dc, fc):
        """
        Estimate the swaption NPV using Black-Scholes formula.
        
        Params:
        -------
        obs_date: datetime.date
            The observation date
        dc: DiscountCurve
            The curve to discount the npv.
        fc: ForwardRateCurve
            The ibor curve to compute the swap rate.
        """
        T = (self.exercise_date - obs_date).days/365
        N = self.irs.nominal
        K = self.irs.fixed_rate
        S = self.irs.swap_rate(dc, fc)
        A = self.irs.annuity(dc)
        dp = (np.log(S/K) + 0.5*self.sigma**2*T)/(self.sigma*np.sqrt(T))
        dm = (np.log(S/K) - 0.5*self.sigma**2*T)/(self.sigma*np.sqrt(T))
        return N*A*(S*norm.cdf(dp)-K*norm.cdf(dm))
    
    def payoffMC(self, obs_date, dc, fc, n_scenarios=100000):
        """
        Estimate the swaption NPV with Monte Carlo Simulation.
        
        Params:
        -------
        obs_date: datetime.date
            Observation date
        dc: DiscountCurve
            The curve to discount the npv.
        fc: ForwardRateCurve
            The libor curve to compute the swap rate.
        n_scenarios: int
            Number of Monte Carlo experiment to simulate.
        """
        T = (self.exercise_date - obs_date).days/365
        payoffs = []
        S0 = self.irs.swap_rate(dc, fc)
        for _ in range(n_scenarios):
            S = S0 * np.exp(-self.sigma**2/2*T + self.sigma*np.random.normal()*np.sqrt(T))
            payoff = self.irs.nominal*max(0, S - self.irs.fixed_rate)*self.irs.annuity(dc)
            payoffs.append(payoff)
        payoff = np.mean(payoffs)
        interval = 1.96*np.std(payoffs)/np.sqrt(n_scenarios)
        return payoff, interval

class CreditCurve:
    """
    CreditCurve: a class to manage credit curves.

    Attributes:
    -----------
    obs_date: datetime.date
        Observation date.
    pillar_date: list of datetime.date
        List of dates that forms the pillars of the curve.
    ndps: list of floats
        List of non-default probabilities.
    """    
    def __init__(self, obs_date, pillar_dates, ndps):
        self.obs_date = obs_date
        self.pillar_dates = pillar_dates
        self.pillar_days = [obs_date] + [(pd - pillar_dates[0]).days for pd in pillar_dates]
        self.ndps = [1] + ndps
        
    def ndp(self, d):
        """
        Interpolate non-default probability at arbitrary dates.

        Params:
        -------
        d: datatime.date
            The date of the interpolation.
        """
        d_days = (d - self.obs_date).days
        if d_days < self.obs_date or d_days > self.pillar_days[-1]:
            print ("Cannot extrapolate survival probability.")
            return None
        return numpy.interp(d_days, self.pillar_days, self.ndps)
    
    def hazard(self, d):
        """
        Compute the annualized hazard rate.

        Params:
        -------
        d: datetime.date
            The date at which the hazard rate is computed.
        """
        ndp_1 = self.ndp(d)
        ndp_2 = self.ndp(d + relativedelta(days=1))
        delta_t = 1.0 / 365.0
        h = -1.0 / ndp_1 * (ndp_2 - ndp_1) / delta_t
        return h
    
class CreditDefaultSwap:
    """
    CreditDefaultSwap: a class to valuate Credit Default Swaps

    Attributes:
    -----------
    nominal: float
        Nominal of the swap.
    start_date: datetime.date
        Starting date of the contract.
    maturity: std
        Maturity of the swap.
    fixed_spread: float
        The spread associated to the premium leg.
    tenor: str
        Tenor of the premium leg, default is 3m.
    recovery: float
        Recovery parameter in case of default, default value is 40%
    """    
    def __init__(self, nominal, start_date, maturity, fixed_spread,
                 tenor="3m", recovery=0.4):
        self.notional = notional
        self.payment_dates = generate_dates(start_date, maturity, tenor)
        self.fixed_spread = fixed_spread
        self.recovery = recovery

    def npv_premium_leg(self, dc, cc):
        """
        Valuate the premium leg.

        Params:
        -------
        dc: DiscountCurve 
            The curve to discount the NPV.
        cc: CreditCurve
            The curve to extract the default probabilities.
        """
        npv = 0
        for i in range(1, len(self.payment_dates)):
            npv += (dc.df(self.payment_dates[i]) *
                    cc.ndp(self.payment_dates[i]))
        return self.fixed_spread * npv * self.notional

    def npv_default_leg(self, dc, cc):
        """
        Valuate the default leg.

        Params:
        -------
        dc: DiscountCurve 
            The curve to discount the NPV.
        cc: CreditCurve
            The curve to extract the default probabilities.
        """
        npv = 0
        d = self.payment_dates[0]
        while d <= self.payment_dates[-1]:
            npv += dc.df(d) * (
                   cc.ndp(d) -
                   cc.ndp(d + relativedelta(days=1)))
            d += relativedelta(days=1)
        return npv * self.notional * (1 - self.recovery)

    def npv(self, dc, cc):
        """
        Valuate the CDS.

        Params:
        -------
        dc: DiscountCurve 
            The curve to discount the NPV.
        cc: CreditCurve
            The curve to extract the default probabilities.
        """
        return self.npv_default_leg(dc, cc) - self.npv_premium_leg(dc, cc)

    def breakevenRate(self, dc, cc):
        """
        Compute the swap breakeven.

        Params:
        -------
        dc: DiscountCurve 
            The curve to discount the NPV.
        cc: CreditCurve
            The curve to extract the default probabilities.
        """
        num = self.npv_default_leg(dc, cc)
        den = self.npv_premium_leg(dc, cc)/self.fixed_spread
        return num/den

class ExpDefault(rv_continuous):
    """
    ExpDefault - class that describes lambda * exp(-lambda*x) distribution, inherits from rv_continuous.
    
    Params:
    -------
    lambda: float
        Lambda parameter of the distribution.
    """
    def __init__(self, l):
        super().__init__()
        self.ulim = 100
        self.l = l
        self.ppf_func = self.prepare_ppf()

    def _cdf(self, x):
        """
        _cdf - reimplement the same method from parent class.

        Params:
        -------
        x: float or numpy.array
            Values where to compute the distribution CDF.
        """
        x[x < 0] = 0
        return (1 - np.exp(-self.l*x))

    def _pdf(self, x):
        """
        _pdf - reimplement the same method from parent class.

        Params:
        -------
        x: float or numpy.array
            Values where to compute the distribution PDF.
        """
        x[x < 0] = 0
        return self.l*np.exp(-self.l*x)

    def _ppf(self, x):
        """
        _ppf - reimplement the same method from parent class.

        Params:
        -------
        x: float or numpy.array
            Values where to compute the distribution PPF.
        """
        return self.ppf_func(x)
  
    def prepare_ppf(self):
        """
        prepare_ppf - utility function to ease ppf calculation.

        Params:
        -------
        """
        xs = np.linspace(0, self.ulim, 10000001)
        cdf = self.cdf(xs)/self.cdf(xs[-1])
        func_ppf = interp1d(cdf, xs, fill_value='extrapolate')
        return func_ppf
    
class BasketDefaultSwaps:
    """
    BasketDefaultSwaps: a class to valuate Basket Default Swaps

    Attributes:
    -----------
    nominal: float
        Notional of the swap.
    N: int
        umber of reference entities underlying the BDS.
    hazard_rate: float
        Hazard rate for default probabilities.
    rho: float
        Correlation factor for the defaults.    
    start_date: datetime.date
        Starting date of the contract.
    maturity: str
        Maturity of the swap.
    spread: float
        The spread associated to the premium leg.
    tenor: str
        Tenor of the premium leg, default is 3m.
    recovery: float
        Recovery parameter in case of default, default value is 40%
    """    
    def __init__(self, nominal, N, hazard_rate, rho, start_date, maturity,
                 spread, tenor=3, recovery=0.4):
        self.cds = CreditDefaultSwap(nominal, start_date, maturity,
                                     spread, tenor, recovery)
        self.Q = ExpDefault(l=hazard_rate)
        self.N = N
        self.rho = rho
        self.cc = None

    def credit_curve(self, pillars, n_defaults):
        """
        Compute the credit curve needed for the BDS valuation.

        Params:
        -------
        pillars: list(datetime.date) 
            Pillars to determine non-default probabilities.
        n_defaults: int
            Number of defaults required by the BDS.
        """
        simulations = 100000
        cov = np.ones(shape=(self.N, self.N))*self.rho
        np.fill_diagonal(cov, 1)
        mean = np.zeros(self.N)
        mv = ss.multivariate_normal(mean=mean, cov=cov)
        x = mv.rvs(size=simulations)
        x_unif = norm.cdf(x)
        default_times = self.Q.ppf(x_unif)

        Ts = [(p-pillars[0]).days/360 for p in pillars]
        ndps = []
        for t in Ts:
            b = np.count_nonzero(default_times<=t, axis=1)
            ndps.append(1 - len(b[b>=n_defaults])/simulations)
        self.cc = CreditCurve(pillars, ndps)

    def npv(self, dc):
        """
        Compute the npv of the BDS.

        Params:
        -------
        dc: DiscountCurve 
            Discount curve to valuate the contract.
        """
        if self.cc is None:
            print ("Need to call credit_curve method first !")
            return None
        return self.cds.npv(dc, self.cc)
  
    def breakeven(self, dc):
        """
        Compute the breakeven of the BDS.

        Params:
        -------
        dc: DiscountCurve 
            Discount curve to valuate the contract.
        """
        return self.cds.breakevenRate(dc, self.cc)
    
class BasketDefaultSwapsOneFactor:
    """
    BasketDefaultSwapsOneFactor: a class to valuate Basket Default Swaps according to the Gaussian One Factor Model

    Attributes:
    -----------
    nominal: float
        Notional of the swap.
    N: int
        Reference entities underlying the BDS.
    rho: float
        Correlation factor for the defaults
    start_date: datetime.date
        Starting date of the contract.
    spread: float
        The spread associated to the premium leg.
    maturity: str
        Maturity of the swap.
    tenor: int
        Tenor of the premium leg in months, default is 3.
    recovery: float
        Recovery parameter in case of default, default value is 40%
    """
    def __init__(self, nominal, N, rho, start_date, maturity,
                 spread, tenor=3, recovery=0.4):
        self.N = N
        self.rho = rho
        self.cds = CreditDefaultSwap(nominal, start_date, maturity,
                                     spread, tenor, recovery)

    def one_factor_model(self, M, f, Q_dates, Q, dc, ndefaults):
        """
        Estimate a quantity value according to the One Factor Model

        Params:
        ------- 
        M: int
            Market value.
        f: user defined function
            Function returning the quantity to estimate.
        Q_dates: list(datetime.date)
            List of dates with known default probability.
        Q: list(float)
            Default probabilities.
        dc: DiscountCurve
            Discount Curve object,
        ndefaults: int
            Number of "required" default in the basket definition.
        """
        P = norm.cdf((norm.ppf(Q) - np.sqrt(self.rho)*M)/np.sqrt(1-self.rho))
        b = binom(self.N, P)
        S = 1 - (1 - b.cdf(ndefaults-1))
        cc = CreditCurve(Q_dates, S)
        return f(dc, cc)*norm.pdf(M)
            
    def breakeven(self, Q_dates, Q, dc, ndefaults):
        """
        Compute the breakeven of the BDS.

        Params:
        -------
        Q_dates: list(datetime.date)
            List of dates with known default probability.
        Q: list(float)
            Default probabilities.
        dc: DiscountCurve
            Discount Curve object,
        ndefaults: int
            Number of "required" default in the basket definition.
        """        
        s = quad(self.one_factor_model, -np.inf, np.inf, 
                 args=(self.cds.breakevenRate, Q_dates, Q, dc, ndefaults))
        return s[0]
    
    def npv(self, Q_dates, Q, dc, ndefaults):
        """
        Compute the npv of the BDS.

        Params:
        -------
        Q_dates: list(datetime.date)
            List of dates with known default probability.
        Q: list(float)
            Default probabilities.
        dc: DiscountCurve
            Discount Curve object,
        ndefaults: int
            Number of "required" default in the basket definition.
        """
        s = quad(self.one_factor_model, -np.inf, np.inf, 
                 args=(self.cds.npv, Q_dates, Q, dc, ndefaults))
        return s[0]        
             
def var_continuous(f, alpha=0.95):
    """
    var_continuous: computes VaR at a specified confidence level, given a continuous loss distribution
    
    Params:
    -------
    f: scipy.stats.rv_continuous
        Continuous distribution representing the portfolio losses.
    alpha: float
        Confidence level for VaR calculation.
    """
    return -f.ppf(1-alpha)

def es_continuous(f, alpha=0.95):
    """
    es_continuous: computes Expected Shortfall at a specified confidence level, given a continuous loss distribution
    
    Params:
    -------
    f: scipy.stats.rv_continuous
        Continuous distribution representing the portfolio losses.
    alpha: float
        Confidence level for ES calculation.
    """
    def integrand(x, f):
        return f.ppf(x)
    alpha = 1-alpha
    I = quad(integrand, 0, alpha, args=(f,))
    return -1/alpha*I[0]
  
def generate_returns(df, N=100000):
    data = df.reset_index()
    return data.loc[np.random.choice(range(len(data)), N)]

def var_discrete(df, alpha=0.95):
    """
    var_discrete: computes VaR at a specified confidence level, given a discrete loss distribution as a DataFrame
    
    Params:
    -------
    df: pandas.DataFrame
        DataFrame containing the loss descrete distribution.
    alpha: float
        Confidence level for VaR calculation.
    """
    alpha = 1-alpha
    return -np.percentile(df, alpha*100)
    
def es_discrete(df, alpha=0.95):
    """
    es_continuous: computes Expected Shortfall at a specified confidence level, given a discrete loss distribution
    
    Params:
    -------
    df: pandas.DataFrame
        DataFrame containing the loss descrete distribution.
    alpha: float
        Confidence level for ES calculation.
    """
    alpha = 1-alpha
    var = np.percentile(df, alpha*100)	
    return -df[df<=var].mean()
