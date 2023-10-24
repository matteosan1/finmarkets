import unittest
import pandas as pd
import numpy as np
from numpy.random import normal, seed

from scipy.optimize import minimize

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve, CreditCurve, CreditDefaultSwap, BasketDefaultSwaps

class Test_Credit(unittest.TestCase):
  def test_credit_curve(self):
    obs_date = date(2022, 10, 1)
    cc = CreditCurve(obs_date, [obs_date + relativedelta(years=2)], [0.8])
    surv_prob = cc.ndp(obs_date + relativedelta(years=1))
    hazard = cc.hazard(obs_date + relativedelta(years=1))
    self.assertAlmostEqual(surv_prob, 0.900, places=3)
    self.assertAlmostEqual(hazard, 0.111, places=3)
    #print ("Survival prob: {:.3f}".format(cc.ndp(obs_date + relativedelta(years=1))))
    #print ("Hazard rate:   {:.3f}".format(cc.hazard(obs_date + relativedelta(years=1))))

  def test_cds(self):
    obs_date = date(2023, 10, 1)
    dc_data = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_curve.xlsx")
    start_date = obs_date
    dates = [obs_date + relativedelta(months=i) for i in dc_data['months']]
    dc = DiscountCurve(obs_date, dates, dc_data['dfs'])

    pillars = [obs_date + relativedelta(months=36)]
    credit_curve = CreditCurve(obs_date, pillars, [0.7])
    cds = CreditDefaultSwap(1e6, start_date, "3y", 0.03)
    npv_prem = cds.npv_premium_leg(dc, credit_curve)
    npv_def = cds.npv_default_leg(dc, credit_curve)
    npv = cds.npv(dc, credit_curve)
    self.assertAlmostEqual(npv_prem, 75830.53, places=2)
    self.assertAlmostEqual(npv_def, 180902.20, places=2)
    self.assertAlmostEqual(npv, 105071.66866249059, places=2)
    #print ("NPV premium: {:.2f}".format(cds.npv_premium_leg(dc, credit_curve)))
    #print ("NPV default: {:.2f}".format(cds.npv_default_leg(dc, credit_curve)))
    #print ("NPV:         {:.2f}".format(cds.npv(dc, credit_curve)))

#  def test_bootstrap(self):
#    obs_date = date(2022, 10, 1)
#    start_date = obs_date
#    dc = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_factors_2022-10-05.xlsx")
#    mq = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/cds_quotes.xlsx")
#
#    dates = [obs_date + relativedelta(months=i) for i in dc['months']]
#    discount_curve = DiscountCurve(obs_date, dates, dc['dfs'])
#
#    cdswaps = []
#    pillar_dates = []
#    for i in range(len(mq)):
#      cds = CreditDefaultSwap(1e6, start_date,
#                              "{}m".format(mq.loc[i, 'months']),
#                              mq.loc[i, 'quotes'])
#      cdswaps.append(cds)
#      pillar_dates.append(cds.payment_dates[-1])
#
#    def objective_function(ndps, obs_date, pillar_dates, discount_curve):
#      credit_curve = CreditCurve(obs_date, pillar_dates, ndps)
#      sum_sq = 0
#      for cds in cdswaps:
#        sum_sq += cds.npv(discount_curve, credit_curve)**2
#      return sum_sq
#
#    ndp_guess = [1 for _ in range(len(cdswaps))]
#    bounds = [(0.01, 1) for _ in range(len(cdswaps))]
#
#    r = minimize(objective_function, ndp_guess, bounds=bounds, 
#                 args=(obs_date, pillar_dates, discount_curve))
#    b = np.array([0.90795, 0.80388, 0.70842, 0.47945, 0.29084, 0.06555])
#    self.assertIsNone(np.testing.assert_array_almost_equal(r.x, b, decimal=5))
#    #self.assertAlmostEqual(r.x, [0.90793109, 0.80383824, 0.70835319, 0.47934782, 0.29074158,
#    #                             0.0654834], places=4)
#    #print (r)

  def test_bds(self):
    obs_date = date(2022, 10, 1)
    n_cds = 10
    rho = 0.3
    l = 0.06
    pillar_dates = [obs_date + relativedelta(years=i) for i in range(1, 6)]
    dfs = [1/(1+0.05)**i for i in range(1, 6)]
    dc = DiscountCurve(obs_date, pillar_dates, dfs)

    basket = BasketDefaultSwaps(1, n_cds, l, rho, obs_date, "2y", 0.01, "3m")
    basket.credit_curve(obs_date, pillar_dates, 3)
    self.assertAlmostEqual(basket.npv(dc), 0.07148635053489855, delta=0.002)
    #print (basket.npv(dc))

print ("\nTest Credit")
if __name__ == '__main__':
    unittest.main()
