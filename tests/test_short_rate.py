import unittest
import numpy as np

from datetime import date

from finmarkets.short_rates import vasicek, cir
from finmarkets import generate_dates

class Test_ShortRate(unittest.TestCase):
    def test_vasicek(self):
        r0 = 0.03
        v = vasicek.VasicekModel(0.3, 0.10, 0.03)
        n = 1000

        dates = generate_dates(date(2023, 10, 20), "1y", "1d", "d")
        res = []
        dt = 1/365
        for i in range(n):
            r = v.r_generator(r0, dates, i)
            I = np.sum(r[1:])*dt
            res.append(np.exp(-I))

        self.assertAlmostEqual(v.ZCB("1y", r0), 0.9613, places=4)
        self.assertAlmostEqual(np.mean(res), 0.960668, places=4)
        self.assertAlmostEqual(np.std(res)/np.sqrt(n), 0.00047270450662961863, places=4)
        #print ("Exact Vasicek Price: {:.4f}".format(v.ZCB(T, r0)))
        #print ("MC Price: {:.4f}".format(np.mean(res)))
        #print ("MC Std Error: {:.4f}".format(np.std(res)/np.sqrt(n)))

    def test_cir(self):
        c = cir.CIRModel(0.3, 0.07, 0.03)
        dates = generate_dates(date(2023, 10, 20), "1y", "1m", "m")
        r = c.r_generator(0.01875, dates, 1)
        b = np.array([0.01875,    0.02200044, 0.02243012, 0.02295059, 0.02272823, 0.02507336,
                      0.02303181, 0.02654363, 0.0265666,  0.0281279,  0.02882913, 0.03204859,
                      0.02979111])
        self.assertIsNone(np.testing.assert_array_almost_equal(r, b, decimal=4))
        #print (r)

print ("\nTest Short Rates")
if __name__ == '__main__':
    unittest.main()
