import pandas as pd
import numpy as np
from numpy.random import normal, seed

from scipy.optimize import minimize

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve, CreditCurve, CreditDefaultSwap, BasketDefaultSwaps

print ("\nTest Credit")
print ("-----------")


obs_date = date.today()
cc = CreditCurve(obs_date, [obs_date + relativedelta(years=2)], [0.8])

print ("Survival prob: {:.3f}".format(cc.ndp(obs_date + relativedelta(years=1))))
print ("Hazard rate:   {:.3f}".format(cc.hazard(obs_date + relativedelta(years=1))))

dc_data = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_curve.xlsx")
start_date = obs_date
dates = [obs_date + relativedelta(months=i) for i in dc_data['months']]
dc = DiscountCurve(obs_date, dates, dc_data['dfs'])

pillars = [obs_date + relativedelta(months=36)]
credit_curve = CreditCurve(obs_date, pillars, [0.7])
cds = CreditDefaultSwap(1e6, start_date, "3y", 0.03)

print ("NPV premium: {:.2f}".format(cds.npv_premium_leg(dc, credit_curve)))
print ("NPV default: {:.2f}".format(cds.npv_default_leg(dc, credit_curve)))
print ("NPV:         {:.2f}".format(cds.npv(dc, credit_curve)))

dc = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_factors_2022-10-05.xlsx")
mq = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/cds_quotes.xlsx")

dates = [obs_date + relativedelta(months=i) for i in dc['months']]
discount_curve = DiscountCurve(obs_date, dates, dc['dfs'])

cdswaps = []
pillar_dates = []
for i in range(len(mq)):
  cds = CreditDefaultSwap(1e6, start_date,
                          "{}m".format(mq.loc[i, 'months']),
                          mq.loc[i, 'quotes'])
  cdswaps.append(cds)
  pillar_dates.append(cds.payment_dates[-1])

def objective_function(ndps, obs_date, pillar_dates, discount_curve):
  credit_curve = CreditCurve(obs_date, pillar_dates, ndps)
  sum_sq = 0
  for cds in cdswaps:
      sum_sq += cds.npv(discount_curve, credit_curve)**2
  return sum_sq

ndp_guess = [1 for _ in range(len(cdswaps))]
bounds = [(0.01, 1) for _ in range(len(cdswaps))]

r = minimize(objective_function, ndp_guess, bounds=bounds, 
             args=(obs_date, pillar_dates, discount_curve))
print (r)


n_cds = 10
rho = 0.3
l = 0.06
pillar_dates = [obs_date + relativedelta(years=i) for i in range(1, 6)]
dfs = [1/(1+0.05)**i for i in range(1, 6)]
dc = DiscountCurve(obs_date, pillar_dates, dfs)

basket = BasketDefaultSwaps(1, n_cds, l, rho, obs_date, "2y", 0.01, "3m")
basket.credit_curve(obs_date, pillar_dates, 3)
print (basket.npv(dc))

