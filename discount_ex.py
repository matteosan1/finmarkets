import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets.finmarkets import DiscountCurve, ForwardRateCurve

def discount_factor(year):
    return 1/(1 + 0.1)**year

opt1 = {0:3000, 0.5:500, 1:500, 1.5:500, 2:500, 2.5:500, 3:500}
opt2 = {0:5000, 0.5:350, 1:350, 1.5:350, 2:350, 2.5:350, 3:350}
npv1 = sum([discount_factor(k)*v for k,v in opt1.items()])
npv2 = sum([discount_factor(k)*v for k,v in opt2.items()])
print ("Option1: {:.1f}".format(npv1))
print ("Option2: {:.1f}".format(npv2))

######################

def fv_factor(year):
    return (1 + 0.08)**year

fv = 0
for year in range(11):
    fv += 1000*fv_factor(year)
print ("future value: {:.1f}".format(fv))

######################

today = date(2020, 10, 15)
dates = [date(2021, 1, 15), date(2021, 4, 15), date(2021, 7, 15),
         date(2021, 10, 15), date(2022, 10, 15), date(2023, 10, 15),
         date(2024, 10, 15), date(2025, 10, 15), date(2026, 10, 15),
         date(2027, 10, 15), date(2028, 10, 15), date(2029, 10, 15),
         date(2030, 10, 15), date(2031, 10, 15), date(2032, 10, 15),
         date(2033, 10, 15), date(2034, 10, 15), date(2035, 10, 15),
         date(2036, 10, 15), date(2037, 10, 15), date(2038, 10, 15),
         date(2039, 10, 15), date(2040, 10, 15), date(2041, 10, 15),
         date(2042, 10, 15), date(2043, 10, 15), date(2044, 10, 15),
         date(2045, 10, 15), date(2046, 10, 15), date(2047, 10, 15),
         date(2048, 10, 15), date(2049, 10, 15), date(2050, 10, 15)]
yields = [-0.652548, -0.687966, -0.718319, -0.744011, -0.807362,
          -0.822144, -0.803715, -0.763496, -0.709892, -0.649001,
          -0.585169, -0.521425, -0.459808, -0.401628, -0.347657,
          -0.298283, -0.253620, -0.213593, -0.178005, -0.146578,
          -0.118993, -0.094911, -0.073989, -0.055896, -0.040317,
          -0.026957, -0.015546, -0.005840, 0.002383, 0.009320,
          0.015145, 0.020013, 0.024059]

df_dates = [date(2025, 4, 15), date(2031, 4, 15)]

num_dates = [(d-today).days for d in dates]
target = (today+relativedelta(months=18)-today).days
print (np.interp(target, num_dates, yields))

######################

dfs = []
for i, d in enumerate(dates):
    dfs.append(np.exp(-yields[i]/100*((d-today).days/365)))
dc = DiscountCurve(dates[0], dates[1:], dfs[1:])
print (dc.df(date(2025, 4, 15)))
print (dc.df(date(2031, 4, 15)))

######################

fc = ForwardRateCurve(dates[0], dates, yields)
print ("{:.4f}%".format(fc.forward_rate(date(2021, 10, 15),
                                        date(2031, 10, 15))/100))
