import numpy as np

from scipy.stats import norm

from .dates import generate_dates

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
        self.fix_dates = generate_dates(start_date, maturity, tenor_fix)
        self.float_dates = generate_dates(start_date, maturity, tenor_float)

    def annuity(self, dc):
        """
        Computes the fixed leg annuity

        Params:
        -------
        dc: DiscountCurve
            discount curve object used for the annuity
        """
        a = 0
        for i in range(1, len(self.fix_dates)):
            tau = (self.fix_dates[i]-self.fix_dates[i-1]).days/360
            a += tau*dc.df(self.fix_dates[i])
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

    def bpv(self, dc):
        return 0.0001*self.annuity(dc)

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
        for j in range(1, len(self.float_dates)):
            F = fc.forward_rate(self.float_dates[j], self.float_dates[j-1])
            tau = (self.float_dates[j] - self.float_dates[j-1]).days / 360
            D = dc.df(self.float_dates[j])
            num += F * tau * D
        return num/self.annuity(dc)

class Cap:
    """
    A class to represent cap/floor

    Attributes:
    -----------
    nominal: float
        nominal of the cap
    start_date: datetime.date
        starting date of the contract
    maturity: str
        maturity of the cap
    fixed_rate: float
        rate of the fixed leg of the swap
    tenor: str
        tenor of the cap
    K: float
        strike of the cap
    """    
    def __init__(self, nominal, start_date, maturity, tenor, K):
        self.dates = generate_dates(start_date, maturity, tenor)
        self.K = K

    def caplet_price(self, sigma, fc, dc, start_date, end_date):
        """
        Compute a caplet npv

        Params:
        -------
        sigma: float
            caplet volatility
        fc: ForwardRateCurve
            forward curve object used for forward rate calculation
        dc: DiscountCurve
            discount curve object used for discounting
        start_date: datetime.date
            forward start date of the caplet
        end_date: datetime.date
            end date of the caplet
        """
        tau = (end_date - start_date).days/360
        D = dc.df(end_date)
        F = fc.forward_rate(end_date, start_date)
        Tf = (end_date - dc.pillar_dates[0]).days/360
        v = sigma*np.sqrt(Tf)
        d1 = (np.log(F/self.K)+0.5*v**2)/v
        d2 = (np.log(F/self.K)-0.5*v**2)/v
        return D*(F*norm.cdf(d1)-self.K*norm.cdf(d2))

    def npv(self, sigma, fc, dc, target_price=0, debug=False):
        """
        Compute cap npv

        Params:
        -------
        sigma: float
            cap volatility
        fc: ForwardRateCurve
            forward curve object used for forward rate calculation
        dc: DiscountCurve
            discount curve object used for discounting
        target_price: float
            optional argument for bootstrapping, default value 0
        """
        if debug:
            print (self.dates)
        val = 0
        for i in range(1, len(self.dates)):
            val += self.caplet_price(sigma, fc, dc, 
                                     self.dates[i-1], self.dates[i])
        if debug:
            print (val)
        return val-target_price
    
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
        
    def npvBS(self, obs_date, dc, fc):
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
    
    def npvMC(self, obs_date, dc, fc, n_scenarios=10000, seed=1):
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
        n_scenarios: int (default = 10000)
            number of Monte Carlo experiment to simulate
        seed: int (default = 1)
            seed for the random number generator
        """
        np.random.seed(seed)
        T = (self.exercise_date - obs_date).days/365
        S0 = self.irs.swap_rate(dc, fc)
        S = S0 * np.exp(-self.sigma**2/2*T + self.sigma*np.random.normal(size=n_scenarios)*np.sqrt(T))
        payoffs = self.irs.nominal*np.maximum(0, S - self.irs.fixed_rate)*self.irs.annuity(dc)
        npv = np.mean(payoffs)
        one_sigma = np.std(payoffs)/np.sqrt(n_scenarios)
        return npv, one_sigma
