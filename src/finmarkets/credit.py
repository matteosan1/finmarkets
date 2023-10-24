import numpy as np

from dateutil.relativedelta import relativedelta

from scipy.stats import norm, binom, multivariate_normal
from scipy.optimize import brentq
from scipy.integrate import quad

from .dates import generate_dates, dates_diff
from .finmarkets import PoissonProcess, CreditCurve

def generateCreditCurve(start_date, end_date, tenor, process=PoissonProcess, kwargs={"l":0.1}):
    """
    A function returning a CreditCurve according to a given process.

    Params:
    -------
    start_date: datetime.date
        start date of the credit curve
    end_date: datetime.date
        end date of the credit curve
    tenor: str
        string representing the curve tenor
    process: scipy.stats.rv:continuous 
        distribution of the default process (default: PoissonProcess)
    kwargs: dict
        additional parameters to be passed to the process function
    """
    maturity = f"{round(dates_diff(start_date, end_date), 0)}m"
    pillars = generate_dates(start_date, maturity, tenor)
    proc = process(**kwargs)
    dps = [proc.cdf((p-start_date).days/365) for p in pillars]
    ndps = [1-dp for dp in dps]
    cc = CreditCurve(start_date, pillars, ndps)
    return cc

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

    def yield_to_maturity(self, dc):
        def obj(y, pv):
            val = 0
            for i in range(1, len(self.payment_dates)):
                tau = (self.payment_dates[i] - self.payment_dates[i-1]).days/365
                cpn = self.face_value*self.K*tau
                if i == len(self.payment_dates)-1:
                    cpn += self.face_value
            val += cpn/(1+y*tau)**i
            return val - pv
        pv = self.npv(dc)
        return brentq(obj, -0.3, 1, args=(pv,))

    def duration(self, dc):
        d = 0
        for i in range(1, len(self.payment_dates)):
            tau = (self.payment_dates[i] - self.payment_dates[i-1]).days/365
            cpn = self.face_value*self.K*tau
            if i == len(self.payment_dates)-1:
                cpn += self.face_value*i*tau
            d += cpn*dc.df(self.payment_dates[i])
        return d/self.npv(dc)

    def mod_duration(self, dc):
        d = self.duration(dc)
        y = self.yield_to_maturity(dc)
        m = 1/((self.payment_dates[1] - self.payment_dates[0]).days/365)
        return d/(1+y/m)

    def dv01(self, dc):
        return -self.duration(dc)*self.npv(dc)*0.0001
        
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

class ParAssetSwap:
    """
    A class to represent par asset swaps

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
            tau = (self.payment_dates[i] - self.payment_dates[i-1]).days/365
            npv += dc.df(self.payment_dates[i]) * cc.ndp(self.payment_dates[i]) * tau
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
        self.Q = PoissonProcess(l=hazard_rate)
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
