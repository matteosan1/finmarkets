import unittest, numpy as np, pandas as pd

from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

from finmarkets import DiscountCurve, OvernightIndexSwap, generate_dates, Interval

class Test_Credit(unittest.TestCase):
  def test_ois(self):
    obs_date = date(2023, 10, 1)
    start_date = obs_date
    ois = OvernightIndexSwap(1e6, start_date, Interval("3y"), 0.025)
    
    df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_factors_2022-10-05.xlsx")
    pillars = [obs_date + relativedelta(months=i) for i in df['months']]
    dfs = df['dfs']
    curve = DiscountCurve(obs_date, pillars, dfs)
    self.assertAlmostEqual(ois.npv(curve), -2164.37, places=2)

  def test_bootstrap(self):
    obs_date = date.today() 
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
                                  Interval(f"{data.loc[i, 'months']}M"),
                                  data.loc[i, 'quotes']*0.01)
        swaps.append(swap)
        pillar_dates.append(swap.payment_dates[-1])
      return swaps, pillar_dates

    swaps, pillar_dates = make_swaps(dataframe)
    dfs0 = [0.5 for _ in range(len(swaps))]
    bounds = [(0.01, 10) for _ in range(len(swaps))]
    
    res = minimize(of, dfs0, bounds=bounds, args=(obs_date, pillar_dates, swaps))
    b = np.array([0.99939263, 0.99831201, 0.99692945, 0.99525677, 0.99335846,
                  0.99136706, 0.9892002 , 0.98691644, 0.98469213, 0.98233439,
                  0.98001276, 0.97763573, 0.9638143 , 0.95044587, 0.92527826,
                  0.90036952, 0.8752759 , 0.85061109, 0.8265085 , 0.80234949,
                  0.77842018, 0.75456742, 0.73127098, 0.70888994, 0.64940464,
                  0.58284654, 0.54160531, 0.50880545, 0.44281208, 0.39008517])
    self.assertIsNone(np.testing.assert_array_almost_equal(res.x, b, decimal=4))
    #print (res)

print ("\nTest OvernightIndexSwap")
if __name__ == '__main__':
    unittest.main()
