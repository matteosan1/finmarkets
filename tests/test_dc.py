import unittest, pandas as pd

from datetime import date

from finmarkets import DiscountCurve, TimeInterval

class Test_DiscountCurve(unittest.TestCase):
    def test_df(self):
        df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/master/input_files/discount_factors_2022-10-05.xlsx")

        obs_date = date.today()
        pillars = [obs_date + TimeInterval(i) for i in df['months']]
        dfs = df['dfs'].values
        dc = DiscountCurve(obs_date, pillars, dfs)
        df_date = obs_date + TimeInterval("195d")
        df0 = dc.df(df_date)
        self.assertAlmostEqual(df0, 0.990, places=3)
    
print ("\nTest DiscountCurve")
if __name__ == '__main__':
    unittest.main()
