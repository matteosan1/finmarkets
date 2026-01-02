import global_const

from .dates import generate_dates
from .utils import SwapSide

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
    global global_const

    def __init__(self, nominal, start_date, maturity, fixed_rate, frequency_float, frequency_fix="1y", side=SwapSide.Receiver):
        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.fix_dates = generate_dates(start_date, maturity, frequency_fix)
        self.float_dates = generate_dates(start_date, maturity, frequency_float)
        self.side = side

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
            if global_const.observation_date > self.fix_dates[i]:
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
