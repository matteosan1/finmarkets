import pandas as pd
from finmarkets import InterestRateSwap, ForwardRateCurve, DiscountCurve
from datetime import date
from dateutil.relativedelta import relativedelta

fixed_rate = 0.056
tenor_fix = "6m"
tenor_float = "6m"
N = 100e6
obs_date = date.today()
start_date = obs_date 
discount_data = pd.read_excel('https://github.com/matteosan1/finance_course/raw/master/input_files/discount_curve.xlsx')
dates = [obs_date + relativedelta(months=i) for i in discount_data['months']]
dc = DiscountCurve(obs_date, dates, discount_data.loc[:, 'dfs'])

dates = [obs_date + relativedelta(months=i) for i in [0, 6, 12, 18]]
fr = ForwardRateCurve(obs_date, dates, [0.102, 0.10, 0.105, 0.11])

irs = InterestRateSwap(N, start_date, "18m", fixed_rate, tenor_float, tenor_fix)
print ("IRS NPV: {:.2f} EUR".format(-irs.npv(dc, fr)))


import pandas as pd
from finmarkets import InterestRateSwaption, ForwardRateCurve, DiscountCurve
from datetime import date
from dateutil.relativedelta import relativedelta

obs_date = date.today()
start_date = obs_date + relativedelta(years=1)
exercise_date = start_date
discount_data = pd.read_excel('https://github.com/matteosan1/finance_course/raw/master/input_files/discount_curve.xlsx')
euribor_data = pd.read_excel('https://github.com/matteosan1/finance_course/raw/master/input_files/euribor_curve.xlsx')


dates = [obs_date + relativedelta(months=i) for i in discount_data['months']]
dc = DiscountCurve(obs_date, dates, discount_data.loc[:, 'dfs'])

dates = [obs_date + relativedelta(months=i) for i in euribor_data['months']]
fr = ForwardRateCurve(obs_date, dates, euribor_data['rates'])

sigma = 0.15
rate = 0.04
swaption = InterestRateSwaption(1e6, start_date, exercise_date,
                                "5y", sigma, rate, "6m")
print("Swaption NPV: {:.0f} EUR".format(swaption.payoffBS(obs_date, dc, fr)))
