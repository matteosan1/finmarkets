import unittest, numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import TermStructure, DiscountCurve, GlobalConst

class Test_ForwardCurve(unittest.TestCase):
    def test_forward_rate(self):
        obs_date = GlobalConst.OBSERVATION_DATE
        pillar_dates = [obs_date,
                        obs_date + relativedelta(months=12),
                        obs_date + relativedelta(months=30)]
        rates = [0.0221, 0.0241, 0.025]

        ts = TermStructure(pillar_dates, rates)
        t1 = obs_date + relativedelta(months=12)
        t2 = obs_date + relativedelta(months=24)
        self.assertAlmostEqual(ts.forward_rate(t1, t2), 0.0253, places=5)

    def test_discount(self):
        obs_date = GlobalConst.OBSERVATION_DATE
        t1 = obs_date + relativedelta(months=3)
        t2 = obs_date + relativedelta(months=9)
        pillar_dates_estr = [obs_date + relativedelta(months=12),
                             obs_date + relativedelta(months=34)]
        estr_dfs = [0.97, 0.72]
        pillar_dates_euribor = [obs_date, 
                                obs_date + relativedelta(months=5),
                                obs_date + relativedelta(months=12)]
        euribor = [0.005, 0.01, 0.015]
        
        estr_curve = DiscountCurve(pillar_dates_estr, estr_dfs) 
        euribor_curve = TermStructure(pillar_dates_euribor, euribor) 

        C = estr_curve.df(t1) * euribor_curve.forward_rate(t1, t2)
        t1_frac, r1 = euribor_curve.interp_rate(t1)
        C_pre2008 = np.exp(-r1*t1_frac) * euribor_curve.forward_rate(t1, t2)

        self.assertAlmostEqual(C, 0.015131, places=5)
        self.assertAlmostEqual(C_pre2008, 0.0152157, places=5)
        #print ("C post 2008: {:.5f}".format(C))
        #print ("C pre 2008: {:.5f}".format(C_pre2008))


print ("\nTest TermStructure")
if __name__ == '__main__':
    unittest.main()
