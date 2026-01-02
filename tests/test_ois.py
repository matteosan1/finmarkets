import unittest, numpy as np, pandas as pd

from datetime import date
from scipy.optimize import minimize

from finmarkets import DiscountCurve, OvernightIndexSwap, TimeInterval, Bootstrap

class Test_Credit(unittest.TestCase):
  def test_ois(self):
    obs_date = date(2023, 10, 1)
    start_date = obs_date
    ois = OvernightIndexSwap(1e6, start_date, "3y", 0.025)
    
    df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_factors_2022-10-05.xlsx")
    pillars = [obs_date + TimeInterval(i) for i in df['maturities']]
    dfs = df['dfs']
    curve = DiscountCurve(obs_date, pillars, dfs)
    self.assertAlmostEqual(ois.npv(curve), -2164.37, places=2)

  def test_bootstrap(self):
    def make_swaps(data):
      start_date = obs_date
      pillar_dates = []
      swaps = []
      for i in range(len(data)):
        swap = OvernightIndexSwap(1e5, start_date, 
                                  data.loc[i, 'maturities'],
                                  data.loc[i, 'quotes']*0.01)
        swaps.append(swap)
        pillar_dates.append(swap.payment_dates[-1])
      return swaps, pillar_dates

    obs_date = date.today() 
    dataframe = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/ois_2022_09_30.xlsx")
    
    swaps, pillar_dates = make_swaps(dataframe)
    dfs0 = [0.5 for _ in range(len(swaps))]
    bounds = [(0.01, 10) for _ in range(len(swaps))]
    
    bootstrap = Bootstrap(obs_date, swaps)
    dfs = bootstrap.run(DiscountCurve)
    
    b = np.array([np.float64(0.9994122206724274), np.float64(0.9982548923656329), np.float64(0.9968954460654532), np.float64(0.9951780994066994), np.float64(0.9932710848604884), np.float64(0.9912725061648486), np.float64(0.9890994293825), np.float64(0.986863314255848), np.float64(0.9845817231720964), np.float64(0.9822202426496266), np.float64(0.9800127670344598), np.float64(0.9776357383336371), np.float64(0.9637495868572207), np.float64(0.9505110314971982), np.float64(0.9252783642600481), np.float64(0.9003696160790231), np.float64(0.8752760022177131), np.float64(0.8506729759288947), np.float64(0.826508704849393), np.float64(0.8023496941513881), np.float64(0.7784203813078728), np.float64(0.7546246621836037), np.float64(0.7312713639043489), np.float64(0.7088903102008945), np.float64(0.6494049610454333), np.float64(0.5828467999258323), np.float64(0.5416055172301514), np.float64(0.5088374705486085), np.float64(0.44281032425271877), np.float64(0.39010483873267177)])
    self.assertIsNone(np.testing.assert_array_almost_equal(dfs, b, decimal=4))

print ("\nTest OvernightIndexSwap")
if __name__ == '__main__':
    unittest.main()
