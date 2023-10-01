import unittest
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve

class Test_DiscountCurve(unittest.TestCase):
    def test_df(self):
        df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/master/input_files/discount_factors_2022-10-05.xlsx")

        obs_date = date(2023, 10, 1)
        pillars = [obs_date + relativedelta(months=i) for i in df['months']]
        dfs = df['dfs'].values
        dc = DiscountCurve(obs_date, pillars, dfs)
        df_date = obs_date + relativedelta(days=195)
        df0 = dc.df(df_date)
        self.assertAlmostEqual(df0, 0.9903, places=4)
        #print ("discount factor at {}: {:.4f}".format(df_date, df0))
    
print ("\nTest DiscountCurve")
if __name__ == '__main__':
    unittest.main()
