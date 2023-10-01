import unittest
import numpy as np
import matplotlib.pyplot as plt

from stochastic import LogNormalEv

class Test_Stocha(unittest.TestCase):
    def test_call_price(self):
        process = LogNormalEv(100, -0.01, 0.05, 1, 350)
        St = process.simulate()
        #plt.plot(St)
        #plt.show()

        S = 100
        K = 105
        mu = 0.01
        sigma = 0.1
        r = 0.01
        T = 1
        scenarios = 10000

        process = LogNormalEv(S, mu, sigma, T, 1)
        St = np.array([process.simulate()[-1] for _ in range(scenarios)])
        C_price = np.mean(np.maximum(St-K, 0))*np.exp(-r*T)
        self.assertAlmostEqual(C_price, 1.1931454261813674, places=3)
        #print (C_price)

print ("\nTest Stochastic")
if __name__ == '__main__':
    unittest.main()
