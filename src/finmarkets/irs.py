import numpy as np

from scipy.stats import norm
from enum import IntEnum

from .dates import generate_dates

Side = IntEnum("Side", {"Receiver":1, "Payer":-1})
CapFloorType = IntEnum("CapFloorType", {"Cap":1, "Floor":-1})

# CONTROLLARE SIDE FIXME
class FRA:
    """
    A class to represent Forward Rate Agreements.
    The valuation of the contract follows Brigo-Mercurio formula.

    Attributes:
    -----------
    today: datetime.date
        princing date
    notional: float
        notional of the swap
    fixing_date: datetime.date
         fixing date of the contrace
    maturity: str
        maturity of the FRA
    fixed_rate: float
        fixed rate to exchange
    """
  def __init__(self, today, nominal, fixing_date, maturity, fixed_rate):
    self.t = today
    self.T = fixing_date
    self.S = maturity
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
    maturity: str
        maturity of the swap.
    fixed_rate: float
        rate of the fixed leg of the swap
    side: Side
        Payer or Receiver type, default Receiver
    """
    def __init__(self, nominal, start_date, maturity, fixed_rate, side=Side.Receiver):
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
        return self.side(*self.npv_floating(dc) - self.npv_fixed(dc))

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
    frequency_float: str
        tenor of the float leg
    frequency_fix: str
        tenor of the fixed leg. default value is 1 year
    side: Side
        define the Payer or Receiver nature of the swap, default Receiver
    """    
    def __init__(self, nominal, start_date, maturity,
                 fixed_rate, frequency_float, frequency_fix="12m", side=Side.Receiver):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.fix_dates = generate_dates(start_date, maturity, frequency_fix)
        self.float_dates = generate_dates(start_date, maturity, frequency_float)
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
        for i in range(1, len(self.dates)):
            fras.append(FRA(self.dates[0], self.N, self.dates[i-1], self.dates[i], self.K))
            
        vals = [self.side*f.npv(dc) for f in fras]
        return sum(vals), vals

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
        num = dc.df(self.dates[0]) - dc.df(self.dates[-1])
        for i in range(1, len(self.dates)):
            tau = (self.dates[i]-self.dates[i-1]).days/360
            den += dc.df(self.dates[i])*tau
        return num/den

    def swap_price_tangent_mode_manual(self, float_rates_dot, zero_rate_dot):
        """
        Compute the sensitivity to rate using tangent method

        Params:
        -------
        float_rates_dot: float
            delta rate to apply to interest rate
        fixed_rates_dot: float
            delta rate to apply to interest rate
        """
        fixed_pv = 0.0
        fixed_pv_dot = 0.0

        for i in range(len(self.fixed_t)):
            fixed_pv += self.notional * self.fixed_rate * self.fixed_tau[i] * np.exp(-self.zero_rate*self.fixed_t[i])
            fixed_pv_dot += -self.fixed_t[i] * self.notional * self.fixed_rate * self.fixed_tau[i] * np.exp(-self.zero_rate*self.fixed_t[i]) * zero_rate_dot

        float_pv = 0.0;
        float_pv_dot = 0.0

        for j in range(len(self.float_t)):
            float_pv += self.notional * (self.float_rates[j]) * self.float_tau[j] * np.exp(-self.zero_rate*self.float_t[j]) # df = exp(-z*t)
            float_pv_dot += self.notional * self.float_tau[j] * np.exp(-self.zero_rate*self.float_t[j]) * float_rates_dot[j]
            float_pv_dot += -self.float_t[j] * self.notional * (self.float_rates[j]) * self.float_tau[j] * np.exp(-self.zero_rate*self.float_t[j]) * zero_rate_dot

        swap_pv = self.side * (fixed_pv - float_pv)
        swap_pv_dot = self.side * (fixed_pv_dot - float_pv_dot)
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
    maturity: str
        maturity of the cap
    fixed_rate: float
        rate of the fixed leg of the swap
    type: CapFloorType
        indicate if it is a caplet or a floorlet
    """    
    def __init__(self, nominal, start_date, maturity,
                 fixed_rate, type=CapFloorType.Cap):
        self.N = nominal
        self.start = start
        self.end = start + relativedelta(months=maturity_from_str(maturity, "m"))
        self.K = fixed
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
    maturity: str
        maturity of the cap
    fixed_rate: float
        rate of the fixed leg of the swap
    tenor: str
        tenor of the cap
    K: float
        strike of the cap
    """    
    def __init__(self, nominal, start_date, maturity, tenor, K, type=CapFloorType.Cap):
        self.dates = generate_dates(start_date, maturity, tenor)
        self.K = K
        self.type = type
        self.maturity = maturity
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
