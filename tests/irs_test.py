import pandas as pd
import numpy as np
from numpy.random import normal, seed

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve, ForwardRateCurve, InterestRateSwap, InterestRateSwaption

print ("\nTest IRS")
print ("--------")

obs_date = date.today()
discount_data = pd.read_excel('https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_factors_2022-10-05.xlsx')
euribor_data = pd.read_excel('https://github.com/matteosan1/finance_course/raw/develop/input_files/euribor_curve.xlsx', sheet_name='EURIBOR3M')

dates = [obs_date + relativedelta(months=i) for i in discount_data['months']]
dc = DiscountCurve(obs_date, dates, discount_data.loc[:, 'dfs'])

dates = [obs_date + relativedelta(months=i) for i in euribor_data['months']]
fr = ForwardRateCurve(obs_date, dates, euribor_data.loc[:, 'rates']*0.01)

print ("df value: {:.3f}".format(dc.df(obs_date + relativedelta(months=15))))
print ("rate:     {:.3f}".format(fr.forward_rate(obs_date + relativedelta(months=15), 
                                       obs_date + relativedelta(months=22))))

start_date = obs_date + relativedelta(months=1)
nominal = 1e6
fixed_rate = 0.023
tenor = "3m"
maturity = "4y"

irs = InterestRateSwap(nominal, start_date, maturity, fixed_rate, tenor)
print ("NPV: {:.2f} EUR".format(irs.npv(dc, fr)))

start_date = obs_date + relativedelta(years=1)
exercise_date = start_date
volatility = 0.15
swaption = InterestRateSwaption(nominal, start_date, exercise_date, maturity, 
                                volatility, fixed_rate, tenor)

price_mc, interval = swaption.payoffMC(obs_date, dc, fr)
print ("MC: {:.2f} +- {:.2f}".format(price_mc, interval))

price_bs = swaption.payoffBS(obs_date, dc, fr)
print ("BS: {:.2f}".format(price_bs))
