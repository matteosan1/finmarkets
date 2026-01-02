import unittest, pandas as pd, numpy as np

from finmarkets.ml.pca import PCAWrapper

class Test_ML(unittest.TestCase):
    def test_pca(self):
        np.random.seed(123)
        x = 5*np.random.rand(100)
        y = 2*x + 1 + np.random.rand(100)
        X = pd.DataFrame(np.vstack([x, y]).T, columns=['x', 'y'])
        pcaw = PCAWrapper(X, normalize=True)
        pcaw.fit()
        cps = pcaw.components()
        ev = pcaw.explained_var()
        self.assertTrue(np.allclose(cps.values, np.array([[0.707107, -0.707107], [0.707107, 0.707107]]), rtol=1e-5, atol=1e-8))
        self.assertTrue(np.allclose(ev, np.array([0.99662716, 0.00337284]), rtol=1e-5, atol=1e-8))

print ("\nTest ML")
if __name__ == '__main__':
    unittest.main()
