import numpy as np

try:
    import tensorflow as tf
except:
    print ("Tensorflow not installed, few features won't be available")

from scipy.stats import norm
from scipy.optimize import newton
from enum import IntEnum

from .dates import generate_dates, Interval, IntervalType
from finmarkets.options.vanilla import OptionType

SwapSide = IntEnum("SwapSide", {"Receiver":1, "Payer":-1})
CapFloorType = IntEnum("CapFloorType", {"Cap":1, "Floor":-1})

class FRA:
    """
    A class to represent Forward Rate Agreements.
    The valuation of the contract follows Brigo-Mercurio formula.

    Attributes:
    -----------
    start_date: datetime.date
        start date of the FRA
    notional: float
        notional of the swap
    fixing_date: Interval
         fixing date of the contrace
    maturity: Interval
        maturity of the FRA
    fixed_rate: float
        fixed rate to exchange
    """
    def __init__(self, start_sate, nominal, fixing_date, maturity, fixed_rate):
        self.t = start_sate
        self.T = fixing_date + start_sate
        self.S = maturity + start_sate
        self.N = nominal
        self.K = fixed_rate

    def npv(self, dc):
        tau = (self.S - self.T).days/360
        P_tS = dc.df(self.S)
        P_tT = dc.df(self.T)
        return self.N*(P_tS*tau*self.K - P_tT + P_tS)

class OvernightIndexSwap:
    """
    A class to represent O/N swaps

    Attributes:
    -----------
    notional: float
        notional of the swap
    start_date: datetime.date
         start date of the contract
    maturity: Interval
        maturity of the swap.
    fixed_rate: float
        rate of the fixed leg of the swap
    side: Side
        Payer or Receiver type, default Receiver
    """
    def __init__(self, nominal, start_date, maturity, fixed_rate, side=SwapSide.Receiver):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.payment_dates = generate_dates(start_date, maturity)
        self.side = side
      
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
        return self.side*(self.npv_floating(dc) - self.npv_fixed(dc))

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
    maturity: Interval
        maturity of the swap.
    fixed_rate: float
        rate of the fixed leg of the swap
    frequency_float: Interval
        tenor of the float leg
    frequency_fix: Interval
        tenor of the fixed leg. default value is 1 year
    side: Side
        define the Payer or Receiver nature of the swap, default Receiver
    """    
    def __init__(self, nominal, start_date, maturity,
                 fixed_rate, frequency_float, frequency_fix=Interval(IntervalType.Annual), 
                 side=SwapSide.Receiver):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.fix_dates = generate_dates(start_date, maturity.add_to(start_date), frequency_fix)
        self.float_dates = generate_dates(start_date, maturity.add_to(start_date), frequency_float)
        self.side = side

    def npv_with_FRA(self, dc):
        """
        Compute the npv of the swa[ assuming it is a collection of FRA's

        Params:
        -------
        dc: DiscountCurve
            discount curve to be used in the calculation
        """
        fras = []
        for i in range(1, len(self.fix_dates)):
            fras.append(FRA(self.fix_dates[0], self.nominal, self.fix_dates[i-1], self.fix_dates[i], self.fixed_rate))
            
        vals = [self.side*f.npv(dc) for f in fras]
        return sum(vals), vals

    def annuity(self, dc, current_date=None):
        """
        Computes the fixed leg annuity

        Params:
        -------
        dc: DiscountCurve
            discount curve object used for the annuity
        current_date: datetime.date
            calculation date for the annuity, if None it is set to the IRS start_date
        """
        if current_date is None:
            current_date = self.fix_dates[0]

        a = 0
        for i in range(1, len(self.fix_dates)):
            if current_date > self.fix_dates[i]:
                continue
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
        return self.side*self.nominal*(self.fixed_rate - S)*A

    def bpv(self, dc):
        """
        Compute the bpv sensitivity of the IRS

        Params:
        -------
        dc: DiscountCurve
            discount curve to apply in the calculation
        """
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

    def swap_rate_single_curve(self, dc):
        """
        Compute the swap rate of the IRS in the single curve framework

        Params:
        -------
        dc: DiscountCurve
            discount curve object used for swap rate calculation
        """        
        den = 0
        num = dc.df(self.fix_dates[0]) - dc.df(self.fix_dates[-1])
        for i in range(1, len(self.fix_dates)):
            tau = (self.fix_dates[i]-self.fix_dates[i-1]).days/360
            den += dc.df(self.fix_dates[i])*tau
        return num/den

class Swap:
    def __init__(self, notional, fixed_rate, tau, terms, float_rates, zero_rate):
        self.N = notional
        self.K = fixed_rate
        self.tau = tau
        self.terms = terms
        self.float_rates = float_rates
        self.r = zero_rate
        
    def swap_price(self, float_rates_dot=0, r_dot=0):
        fixed_pv = tf.Variable(0.0)
        r = tf.Variable(self.r, name="r")
        float_pv = tf.Variable(0.0)
        float_rates = tf.Variable(self.float_rates, name="float_rates")

        with tf.GradientTape(persistent=True) as tape:
            for i in range(len(self.terms)):
                fixed_pv = fixed_pv + self.N*self.K*self.tau*tf.math.exp(-r*self.terms[i])
                if r_dot != 0:
                    fixed_pv_dot = r_dot*tape.gradient(fixed_pv, r)
                else:
                    fixed_pv_dot = 0

        with tf.GradientTape(persistent=True) as tape:
            for j in range(len(self.terms)):
                float_pv = float_pv + self.N*float_rates[j]*self.tau*tf.math.exp(-r*self.terms[j])
        float_pv_dot = 0
        if r_dot != 0:
            float_pv_dot += r_dot*tape.gradient(float_pv, r)
        if float_rates_dot != 0:
            float_pv_dot += sum(tape.gradient(float_pv, float_rates)*float_rates_dot)

        swap_pv = (fixed_pv - float_pv)
        swap_pv_dot = (fixed_pv_dot - float_pv_dot)

        return swap_pv, swap_pv_dot

    def swap_price_tangent_mode_manual(self, float_rates_dot, r_dot):
        fixed_pv = 0.0
        fixed_pv_dot = 0.0

        for i in range(len(self.terms)):
            fixed_pv += self.N*self.K*self.tau*np.exp(-self.r*self.terms[i])
            fixed_pv_dot += -self.terms[i]*self.N*self.K*self.tau*np.exp(-self.r*self.terms[i])*r_dot

        float_pv = 0.0;
        float_pv_dot = 0.0
        for j in range(len(self.terms)):
            float_pv += self.N*(self.float_rates[j])*self.tau*np.exp(-self.r*self.terms[j])
            float_pv_dot += self.N*self.tau*np.exp(-self.r*self.terms[j])*float_rates_dot
            float_pv_dot += -self.terms[j]*self.N*(self.float_rates[j])*self.tau*np.exp(-self.r*self.terms[j])*r_dot
            
        swap_pv = (fixed_pv - float_pv)
        swap_pv_dot = (fixed_pv_dot - float_pv_dot)
        return swap_pv, swap_pv_dot
    
class CapFloorLet:
    """
    A class to represent cap/floorlet
        
    Attributes:
    -----------
    nominal: float
        nominal of the cap
    start_date: datetime.date
        starting date of the contract
    maturity: Interval
        maturity of the cap
    fixed_rate: float
        rate of the fixed leg of the swap
    type: CapFloorType
        indicate if it is a caplet or a floorlet
    """    
    def __init__(self, nominal, start_date, maturity, fixed_rate, type=CapFloorType.Cap):
        self.N = nominal
        self.start = start_date
        self.end = start_date + maturity #generate_dates(start_date, maturity, maturity)[-1]
        self.K = fixed_rate
        self.type = type

    def npv(self, sigma, dc, fc):
        """
        Compute the npv of the cap/floorlet using BS formula
        
        Params:
        -------
        sigma: float
            volatility of the cap/floorlet
        dc: DiscountCurve
            discount curve
        fc: ForwardRateCurve
            interest rate term structure
        """
        tau = (self.end - self.start).days/360
        D = dc.df(self.end)
        F = fc.forward_rate(self.start, self.end)
        Tf = (self.start - dc.pillar_dates[0]).days/360
        v = sigma*np.sqrt(Tf)
        d1 = (np.log(F/self.K)+0.5*v**2)/v
        d2 = (np.log(F/self.K)-0.5*v**2)/v
        if self.type == CapFloorType.Cap:
            return D*(F*norm.cdf(d1)-self.K*norm.cdf(d2))
        else:
            return D(self.K*norm.cdf(-d2)-F*norm.cdf(-d1))

class CapFloor:
    """
    A class to represent cap/floor
        
    Attributes:
    -----------
    nominal: float
        nominal of the cap
    start_date: datetime.date
        starting date of the contract
    maturity: Interval
        maturity of the cap
    fixed_rate: float
        rate of the fixed leg of the swap
    tenor: Interval
        tenor of the cap
    K: float
        strike of the cap
    """    
    def __init__(self, nominal, start_date, maturity, tenor, K, type=CapFloorType.Cap):
        self.dates = generate_dates(start_date, maturity, tenor)
        self.K = K
        self.type = type
        #self.maturity = maturity
        self.tenor = tenor
        self.N = nominal

    def npv(self, sigma, dc, fc):
        """
        Compute the npv of the cap as a sum of Cap/Floorlet
        
        Params:
        -------
        sigma: float
            flat volatility of the Cap
        dc: DiscountCurve
            discount curve
        fc: ForwardCurve
            interest rate curve
        """
        val = 0
        for i in range(0, len(self.dates)-1):
            caplet = CapFloorLet(self.N, self.dates[i], self.tenor, self.K, self.type)
            val += caplet.npv(sigma, dc, fc)
        return val
    
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
    maturity: Interval
        maturity of the swap
    volatility: float
        swap rate volatility
    fixed_rate: float
        rate of the fixed leg of the swap
    tenor: Interval
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

class SwaptionShortRate:
    def __init__(self, notional, expiry, tenor, strike, model, side=SwapSide.Receiver):
        self.expiry = expiry
        self.tenor = tenor
        self.K = strike
        self.N = notional
        if side == SwapSide.Receiver:
            self.option_type = OptionType.Put
        else:
            self.option_type = OptionType.Call
        self.model = model
        self.terms = np.linspace(expiry+1, expiry+tenor, tenor)

    def rstar(self, guess=0.01):
        def obj(r, model, K, tenor, terms):
            val = 0
            for i, T in enumerate(terms):
                val += model.ZCB(r, T-1, T)*K*tenor
            val += model.ZCB(r, T-1, T)
            return val-1
        return newton(obj, guess, args=(self.model, self.K, self.tenor, self.terms))

    def npv(self, r, r_star):
        val = 0
        for i, T in enumerate(self.terms):
            K_k = self.model.ZCB(r_star, T-1, T)
            val += self.model.ZBO(r, K_k, 0, T-1, T, self.option_type)*K_k*self.tenor
        val += self.model.ZBO(r, K_k, 0, T-1, T, self.option_type)
        return val

  # def jamshidian_decomposition(expiry, tenor, strike, a, sigma):
  #   """
  #   Performs Jamshidian's decomposition to find the critical rate.
  #   """
  #   def integrand(t):
  #     return zero_bond(0, t) * stats.norm.pdf(
  #       (np.log(zero_bond(t, expiry + tenor) / zero_bond(t, expiry)) + (a**2 * tenor) / (2 * sigma**2)) /
  #       (sigma * np.sqrt(tenor) / a)
  #     )
  #   kappa = quad(integrand, 0, expiry)[0]
  #   return forward_swap_rate(expiry, tenor) + (sigma**2 * tenor * kappa) / (2 * a)

