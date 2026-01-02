import unittest, numpy as np, pandas as pd

from datetime import date
from scipy.optimize import minimize

from finmarkets import DiscountCurve, OvernightIndexSwap, TimeInterval, Bootstrap, GlobalConst

class Test_Credit(unittest.TestCase):
	def test_ois(self):
		obs_date = start_date = GlobalConst.OBSERVATION_DATE
		ois = OvernightIndexSwap(1e6, start_date, "3y", 0.025)
		
		df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/discount_factors_2022-10-05.xlsx")
		pillars = [obs_date + TimeInterval(i) for i in df['maturities']]
		dfs = df['dfs']
		curve = DiscountCurve(pillars, dfs)
		self.assertAlmostEqual(ois.npv(curve), -2160.994, places=2)

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

		obs_date = GlobalConst.OBSERVATION_DATE 
		dataframe = pd.read_excel("https://github.com/matteosan1/finance_course/raw/develop/input_files/ois_2022_09_30.xlsx")
		
		swaps, pillar_dates = make_swaps(dataframe)
		dfs0 = [0.5 for _ in range(len(swaps))]
		bounds = [(0.01, 10) for _ in range(len(swaps))]
		
		bootstrap = Bootstrap(swaps)
		dc = bootstrap.run(DiscountCurve)
		b = np.array([1., 0.99939264, 0.99831201, 0.99692946, 0.99525677, 0.99335846,
	 				  0.99136707, 0.9892002,  0.98691645, 0.98469214, 0.9823344,  0.98001277,
 					  0.97763574, 0.96387904, 0.95051103, 0.92527836, 0.90036962, 0.875276,
  					  0.85067298, 0.8265087,  0.80234969, 0.77842038, 0.75462466, 0.73127136,
 					  0.70889031, 0.64940496, 0.5828468,  0.54160552, 0.50883747, 0.44281032, 0.39010484])
		self.assertIsNone(np.testing.assert_array_almost_equal(dc.dfs, b, decimal=4))

print ("\nTest OvernightIndexSwap")
if __name__ == '__main__':
    unittest.main()
