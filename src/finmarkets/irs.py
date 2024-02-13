import numpy as np, tensorflow as tf

from scipy.stats import norm
from scipy.optimize import newton
from enum import IntEnum

from .dates import generate_dates

SwapType = IntEnum("SwapType", {"Receiver":-1, "Payer":1})

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
    type: SwapType
        type of the swap. default is SwapType.Payer
    """
    def __init__(self, nominal, start_date, maturity, fixed_rate, type=SwapType.Payer):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.payment_dates = generate_dates(start_date, maturity)
        self.type = type
      
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
        return self.npv_floating(dc) - self.npv_fixed(dc)*self.type

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
    type: SwapType
        type of the swap, either Receiver or Payer. Default: Receiver
    """    
    def __init__(self, nominal, start_date, maturity,
                 fixed_rate, tenor_float, tenor_fix="12m", type=SwapType.Payer):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.fix_dates = generate_dates(start_date, maturity, tenor_fix)
        self.float_dates = generate_dates(start_date, maturity, tenor_float)
        self.type = type

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
        return self.nominal * (S - self.fixed_rate) * A * self.type

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
        
    def npv_with_delta(self, dc, fc, dr=0, dzero_rate=0):
      fixed_pv = tf.Variable(0.0)
      float_pv = tf.Variable(0.0)
      zero_rate_fix = tf.Variable([dc.rate(d) for d in self.fix_dates], name="zero_rate_fix")
      zero_rate_float = tf.Variable([dc.rate(d) for d in self.float_dates], name="zero_rate_float")
      float_rates = tf.Variable([fc.forward_rate(self.float_dates[i-1], self.float_dates[i]) 
          for i in range(len(self.float_dates))], name="float_rates")

      fixed_pv_dot = 0
      float_pv_dot = 0
      with tf.GradientTape(persistent=True) as tape:
        for i in range(1, len(self.fix_dates)):
          tau = (self.fix_dates[i]-self.fix_dates[i-1]).days/360
          dt = (self.fix_dates[i]-self.fix_dates[0]).days/365
          fixed_pv += self.N*self.K*tau*tf.math.exp(-zero_rate_fix[i]*dt)
        if dzero_rate != 0:
          fixed_pv_dot += dzero_rate*sum(tape.gradient(fixed_pv, zero_rate_fix))

        for i in range(1, len(self.float_dates)):
          tau = (self.float_dates[i]-self.float_dates[i-1]).days/360
          dt = (self.float_dates[i]-self.float_dates[0]).days/365
          float_pv += self.N*float_rates[i]*tau*tf.math.exp(-zero_rate_float[i]*dt)      
        if dzero_rate != 0:
          float_pv_dot += dzero_rate*sum(tape.gradient(float_pv, zero_rate_float))
        if dr != 0:
          float_pv_dot += sum(tape.gradient(float_pv, float_rates)*float_rates_dot)

      swap_pv = self.side*(fixed_pv - float_pv)
      swap_pv_dot = self.side*(fixed_pv_dot - float_pv_dot)

      return swap_pv, swap_pv_dot

    def delta_tangent_mode(self, dc fc, dr, dzero_rate):
      fixed_pv_dot = 0.0
      float_pv_dot = 0.0

      for i in range(1, len(self.fix_dates)):
          tau = (self.fix_dates[i]-self.fix_dates[i-1]).days/360
          dt = (self.fix_dates[i]-self.fix_dates[0]).days/360
          fixed_pv_dot += -dt*self.N*self.K*tau*dc.df(self.fix_dates[i])* dzero_rate

      for i in range(1, len(self.float_dates)):
          tau = (self.float_dates[i]-self.float_dates[i-1]).days/360
          dt = (self.float_dates[i]-self.float_dates[0]).days/360
          float_pv_dot += self.N*tau*dc.df(self.float_dates[i])*dr
          float_pv_dot += -dt*self.N*fc.fowrard_rate(self.float_dates[i-1], self.float_dates[i])*tau*dc.df(self.float_dates[i])*dzero_rate

      return self.side*(fixed_pv_dot - float_pv_dot)


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

class SwaptionShortRate:
    def __init__(self, notional, expiry, tenor, strike, type, model):
        self.expiry = expiry
        self.tenor = tenor
        self.K = strike
        self.N = notional
        self.type = type
        self.model = model

    def annuity(self):
        #terms = np.linspace(int(self.expiry+1), int(self.expiry + self.tenor), int(self.tenor))
        terms = np.linspace(self.expiry+1, self.expiry + self.tenor, self.tenor)
        return sum(self.model.zero_bond(0, t) for t in terms)

    def forward_swap_rate(self):
        """
        Calculates the forward swap rate at expiry for a given tenor.
        Assumes annual payments for simplicity.
        """
        terms = np.linspace(self.expiry+1, self.expiry + self.tenor, self.tenor)
        P0 = self.model.zero_bond(0, self.expiry)
        Pn = self.model.zero_bond(0, self.expiry+self.tenor)
        sum_P = sum(self.model.zero_bond(0, t) for t in terms)
        return (P0 - Pn)/self.annuity()

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

    def jamshidian_decomposition_root_finder(self):
        """
        Performs Jamshidian's decomposition using a root finder.
        """
        terms = np.linspace(self.expiry+1, self.expiry + self.tenor, self.tenor)

        def objective(critical_rate):
          bond_portfolio_price = sum(self.model.zero_bond(0, t)*np.maximum(self.K - critical_rate, 0) for t in terms)
          price_with_critical_rate = bond_portfolio_price + self.N*self.model.zero_bond(0, self.expiry)*np.maximum(critical_rate - self.K, 0)
          return price_with_critical_rate  # Aim for zero market price

        critical_rate = newton(objective, 0.01)
        return self.forward_swap_rate() + (self.model.sigma**2*self.tenor*critical_rate)/(2*self.model.a)

    def npv(self):
        terms = np.linspace(self.expiry+1, self.expiry + self.tenor, self.tenor)
        critical_rate = self.jamshidian_decomposition_root_finder()
        bond_portfolio_price = sum(self.model.zero_bond(0, t)*np.maximum(self.K - critical_rate, 0) for t in terms)
        return bond_portfolio_price + self.type*self.N*self.model.zero_bond(0, self.expiry)*np.maximum(self.type*(critical_rate - self.K), 0)        