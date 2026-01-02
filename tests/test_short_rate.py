import unittest
import numpy as np

from datetime import date

from finmarkets.short_rates import vasicek, cir

class Test_ShortRate(unittest.TestCase):
    def test_vasicek(self):
        r0 = 0.03
        np.random.seed(1)
        v = vasicek.VasicekModel(0.3, 0.10, 0.03)
        n = 100
        T = 1
        steps = 365
        res = np.zeros(shape=(n,))
        dt = 1/steps
        for i in range(n):
            r = v.r(r0, n, T, dt)
            res[i] = np.exp(-np.sum(r[1:])*dt)

        self.assertAlmostEqual(v.ZCB_analytical(r0, 1), 0.9613, places=3)
        self.assertAlmostEqual(np.mean(res), 0.02007, places=3)
        self.assertAlmostEqual(np.std(res)/np.sqrt(n), 0.000348, places=5)

    def test_cir(self):
        np.random.seed(1)
        c = cir.CIRModel(0.3, 0.07, 0.03)
        T = 1
        steps = 12
        r = c.r(0.02, T, steps)
        b = np.array([0.02,0.02323941,0.02360078,0.02405806,0.02376533,0.02607657,
                      0.023956,0.02744587,0.0274176,0.02893966,0.02959878,0.03278726])
        self.assertIsNone(np.testing.assert_array_almost_equal(r, b, decimal=4))


print ("\nTest Short Rates")
if __name__ == '__main__':
    unittest.main()
