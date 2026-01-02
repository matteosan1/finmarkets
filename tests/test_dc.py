import unittest, pandas as pd

from datetime import date

from finmarkets import DiscountCurve, TimeInterval, GlobalConst

class Test_DiscountCurve(unittest.TestCase):
    def test_df(self):
        df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/master/input_files/discount_factors_2022-10-05.xlsx")

        pillars = [GlobalConst.OBSERVATION_DATE + TimeInterval(i) for i in df['months']]
        dfs = df['dfs'].values
        dc = DiscountCurve(pillars, dfs)
        df_date = date.today() + TimeInterval("195d")
        df0 = dc.df(df_date)
        self.assertAlmostEqual(df0, 0.990, places=3)
    
print ("\nTest DiscountCurve")
if __name__ == '__main__':
    unittest.main()
