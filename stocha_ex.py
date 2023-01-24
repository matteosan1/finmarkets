from numpy.random import normal, seed
import numpy as np
S = 100
mu = 0.05
sigma = 0.17
T = 1
steps = 360
seed(1)
wiener = [S]
abm = [S]
gbm = [S]
for i in range(steps):
    wiener.append(wiener[-1] + normal()*np.sqrt(T/steps))
    abm.append(abm[-1] + mu * T/steps + sigma * np.sqrt(T/steps) * normal())
    S = S * np.exp((mu - 0.5 * sigma * sigma) * T/steps + sigma * \
                   np.sqrt(T/steps) * normal())
    gbm.append(S)


from finmarkets import call
T = "1y"
S0 = 110
r = 0.05
sigma = 0.17
dt = 1
K = 105
C0 = call(S0, K, r, sigma, T)
print ("BS call price: {:.2f}".format(C0))

import time
import numpy as np
from numpy.random import normal, seed

t1 = time.time()
S0 = 110
r = 0.05
sigma = 0.17
T = 1
K = 105
values = []
for sim in range(1, 10000):
    seed(sim)
    payoff = []
    for i in range(sim):
        St = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * normal())
        payoff.append(np.exp(-r*T)* max(0, St-K))
    values.append(sum(payoff)/sim)
Cs = []
for i in range(1, len(values)):
    Cs.append((sum(values[:i])/i - C0)/C0)
print ("Elapsed time: ", time.time() - t1)

import numpy as np
import time
from scipy.stats import norm
t1 = time.time()
S0 = 110
r = 0.05
sigma = 0.17
dt = 1
K = 105
Cs = []
for i in range(1, 10000):
    np.random.seed(i)
    Z = norm.rvs(size=i)
    St = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    Cs.append(np.mean(np.exp(-r * T) * np.maximum(St - K, 0)))
n = 1/np.arange(1, 10000)
c = np.cumsum(Cs)*n
print ("Elapsed time: ", time.time() - t1)


import numpy as np
C = np.array([0,0,1])
P = np.array([[.8, .1,.1],[.2,.6,.2],[.25,.25,.5]])
C_grandson = C.dot(P).dot(P)
print (C_grandson[0])



import numpy as np
P = np.array([[.5,.5,0],[.25,.5,.25],[0,.5,.5]])
mu1 = P[1, :]
mu2 = P.dot(P)[1, :]
mu3 = P.dot(P).dot(P)[1, :]
print (mu1)
print (mu2)
print (mu3)

