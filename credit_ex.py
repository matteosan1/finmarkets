import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from finmarkets import generate_dates, CreditDefaultSwap, CreditCurve, DiscountCurve, saveObj
from scipy.optimize import minimize

obs_date = date.today()
cds_quotes = [{'maturity': "12m", 'spread':0.0149},
              {'maturity': "24m", 'spread':0.0165},
              {'maturity': "36m", 'spread':0.0173},
              {'maturity': "69m", 'spread':0.0182},
              {'maturity': "120m", 'spread':0.0183},
              {'maturity': "240m", 'spread':0.0184}]

discount_data = pd.read_excel('https://github.com/matteosan1/finance_course/raw/master/input_files/discount_curve.xlsx')
dates = [obs_date + relativedelta(months=i) for i in discount_data['months']]
dc = DiscountCurve(obs_date, dates, discount_data.loc[:, 'dfs'])

cds_dates = []
creditdefaultswaps = []
for quote in cds_quotes:
    creditdefswap = CreditDefaultSwap(1, obs_date, quote['maturity'], quote['spread'])
    creditdefaultswaps.append(creditdefswap)
    cds_dates.append(creditdefswap.payment_dates[-1])
    
def obj_function(unknown_ndps, obs_date, cds_dates, dc):
    curve_c = CreditCurve(obs_date, cds_dates, unknown_ndps)
    sum_sq = 0.0
    for cds in creditdefaultswaps:
        sum_sq += cds.npv(dc, curve_c) ** 2
    return sum_sq

x0_guess = [0.001 for i in range(len(creditdefaultswaps))]
bounds_credit_curve = [(0.01, 1) for i in range(len(creditdefaultswaps))]
results = minimize(obj_function, x0_guess, bounds=bounds_credit_curve,
                   args=(obs_date, cds_dates, dc))
print (results.x)
credit_curve = CreditCurve(obs_date, cds_dates, results.x)
saveObj("credit_curve.pkl", credit_curve)
saveObj("discount_curve.pkl", dc)


from finmarkets import CreditDefaultSwap, CreditCurve, DiscountCurve, loadObj
from datetime import date

obs_date = date.today()
cds_to_price = [{'nominal': 5000000, 'maturity':"18m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"30m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"42m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"72m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"108m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"132m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"160m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"184m", 'spread': 0.02},
                {'nominal': 5000000, 'maturity':"210m", 'spread': 0.02}]

cc = loadObj("prova.pkl")
dc = loadObj("discount_curve.pkl")
npv_cds_to_price = []
for quote in cds_to_price:
    cds = CreditDefaultSwap(quote['nominal'], obs_date,
                            quote['maturity'], quote['spread'])
    npv_cds_to_price.append(cds.npv(dc, cc))
print (npv_cds_to_price)
