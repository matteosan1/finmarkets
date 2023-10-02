import unittest
import pandas as pd
import numpy as np

from scipy.stats import norm

from finmarkets import var_continuous, es_continuous, var_discrete, es_discrete

class Test_RiskMeasures(unittest.TestCase):
    def test_measures(self):
        df = pd.read_csv("https://raw.githubusercontent.com/matteosan1/finance_course/master/input_files/historical_data.csv", index_col='Date')
        df['P'] = df['aapl']*0.6 + df['nflx']*0.4
        df = df.pct_change()
        df.dropna(inplace=True)

        mu = df.mean() 
        sigma = df.std()

        f = norm(mu['P'], sigma['P'])
        var_c = var_continuous(f, 0.95)
        es_c = es_continuous(f, 0.95)
        self.assertAlmostEqual(var_c, 0.03186, places=5)
        self.assertAlmostEqual(es_c, 0.04037, places=5)
        #print ("1d-95% VaR: {:.4}".format(var_c))
        #print ("1d-95% ES: {:.4}".format(es_c))
        var_d = var_discrete(df, 0.95)
        es_d = es_discrete(df, 0.95)
        self.assertAlmostEqual(var_d, 0.0280, delta=0.002)
        self.assertAlmostEqual(es_d, 0.0454, delta=0.002)
        #print ("1d-95% VaR (discrete): {:.4f}".format(var_d))
        #print ("1d-95% ES (discrete): {:.4f}".format(es_d))

print ("\nTest Risk Measures")
if __name__ == '__main__':
    unittest.main()
