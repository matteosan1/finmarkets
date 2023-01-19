import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from finmarkets import ForwardRateCurve, DiscountCurve

print ("\fTest ForwardRateCurve")
print ("------------------")

obs_date = date.today()
pillar_dates = [obs_date,
                obs_date + relativedelta(months=12),
                obs_date + relativedelta(months=30)]
rates = [0.0221, 0.0241, 0.025]

fc = ForwardRateCurve(obs_date, pillar_dates, rates)
t1 = obs_date + relativedelta(months=12)
t2 = obs_date + relativedelta(months=24)
print ("F({}, {}) = {:.4f}".format(t1, t2, fc.forward_rate(t1, t2)))

t1 = obs_date + relativedelta(months=3)
t2 = obs_date + relativedelta(months=9)
pillar_dates_estr = [obs_date + relativedelta(months=12),
                     obs_date + relativedelta(months=34)]
estr_dfs = [0.97, 0.72]
pillar_dates_euribor = [obs_date, 
                        obs_date + relativedelta(months=5),
                        obs_date + relativedelta(months=12)]
euribor = [0.005, 0.01, 0.015]

estr_curve = DiscountCurve(obs_date, pillar_dates_estr, estr_dfs) 
euribor_curve = ForwardRateCurve(obs_date, pillar_dates_euribor, euribor) 

C = estr_curve.df(t1) * euribor_curve.forward_rate(t1, t2)
t1_frac, r1 = euribor_curve.interp_rate(t1)
C_pre2008 = np.exp(-r1*t1_frac) * euribor_curve.forward_rate(t1, t2)

print ("C post 2008: {:.5f}".format(C))
print ("C pre 2008: {:.5f}".format(C_pre2008))
