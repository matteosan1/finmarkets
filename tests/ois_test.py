import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from finmarkets import DiscountCurve, OvernightIndexSwap, generate_dates

print ("\fTest OvernightIndexSwap")
print ("-----------------------")

obs_date = date.today() 
start_date = obs_date
ois = OvernightIndexSwap(1e6, start_date, "3y", 0.025)

df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_factors_2022-10-05.xlsx")
pillars = [obs_date + relativedelta(months=i) for i in df['months']]
dfs = df['dfs']

curve = DiscountCurve(obs_date, pillars, dfs)

print ("OIS NPV: {:.2f}".format(ois.npv(curve)))

dataframe = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/ois_2022_09_30.xlsx")

def of(dfs, obs_date, pillars, swaps):
  dc = DiscountCurve(obs_date, pillars, dfs)
  val = 0
  for s in swaps:
    val += s.npv(dc)**2
  return val

def make_swaps(data):
  start_date = obs_date
  pillar_dates = []
  swaps = []
  for i in range(len(data)):
    swap = OvernightIndexSwap(1e5, start_date, 
                              "{}M".format(data.loc[i, 'months']),
                              data.loc[i, 'quotes']*0.01)
    swaps.append(swap)
    pillar_dates.append(swap.payment_dates[-1])
  return swaps, pillar_dates

swaps, pillar_dates = make_swaps(dataframe)

dfs0 = [0.5 for _ in range(len(swaps))]
bounds = [(0.01, 10) for _ in range(len(swaps))]

res = minimize(of, dfs0, bounds=bounds, args=(obs_date, pillar_dates, swaps))
print (res)
