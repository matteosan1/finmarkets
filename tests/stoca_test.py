import numpy as np
import matplotlib.pyplot as plt

from stochastic import LogNormalEv

print ("\nTest Stochastic")
print ("------------------")

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
print (C_price)
    
