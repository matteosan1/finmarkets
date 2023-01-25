import unittest
import pandas as pd
import numpy as np
from numpy.random import normal, seed

from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import call, maturity_from_str

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

print ("\nTest European Option")
if __name__ == '__main__':
    unittest.main()
