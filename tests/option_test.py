import unittest
import pandas as pd
import numpy as np
from numpy.random import normal, seed

from datetime import date
from dateutil.relativedelta import relativedelta

from options import call, maturity_from_str, AsianOption

class Test_Options(unittest.TestCase):
  def test_call(self):
    S0 = 107
    r = 0.03
    sigma = 0.12
    ttm = "1y"
    T = maturity_from_str(ttm, "y")
    K = 100
    
    seed(1)
    payoffs = []
    experiments = 100000
    for i in range(experiments):
      St = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * normal())
      payoffs.append(np.exp(-r*T)*max(0, St-K))

    C_MC = np.mean(payoffs)
    cl95 =1.96*np.std(payoffs)/np.sqrt(experiments)
    self.assertAlmostEqual(C_MC, 11.430, places=3)
    self.assertAlmostEqual(cl95, 0.068, places=3)
    #print ("MC BS call price: {:.3f} +- {:.3f} @ 95% confidence level".format(C_MC, cl95))
    C_BS = call(S0, K, r, sigma, ttm)
    self.assertAlmostEqual(C_BS, 11.388, places=3)
    #print ("BS call price: {:.3f}".format(C_BS))

class Test_AsianOptions(unittest.TestCase):
    def test_asiancall(self):
      S0 = 100
      ttm = "1y"
      sigma = 0.2
      r = 0.05
      K = 90
      N = 1000

      prices = np.zeros(3)
      errs = np.zeros(3)

      opt = AsianOption(S0, K, r, sigma, ttm)
      p, e = opt.simulate_naive(N)
      #print ("{:.3f} +- {:.4f}".format(p, e))
      prices[0] = p
      errs[0] = e
      
      p, e = opt.simulate_antithetic(N)
      #print ("{:.3f} +- {:.4f}".format(p, e))
      prices[1] = p
      errs[1] = e
      
      p, e = opt.simulate_control_variate(N)
      #print ("{:.3f} +- {:.4f}".format(p, e))
      prices[2] = p
      errs[2] = e

      res = np.array([27.434, 27.685, 27.889])
      err_res = np.array([1.0573, 0.4783, 0.1474])
      self.assertIsNone(np.testing.assert_array_almost_equal(prices, res, decimal=3))
      self.assertIsNone(np.testing.assert_array_almost_equal(errs, err_res, decimal=3))

print ("\nTest European and Asian Options")
if __name__ == '__main__':
    unittest.main()
