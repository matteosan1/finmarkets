#import pandas as pd
#import numpy as np
#from numpy.random import seed, choice
#from numpy import percentile
#from finmarkets import generate_returns, var_discrete, es_discrete
#
#df = pd.read_csv("https://raw.githubusercontent.com/matteosan1/finance_course/develop/input_files/historical.csv", index_col='date')
#w = np.array([0.4, 0.25, 0.35])
#df['P'] = df[['FOX', 'CBS', 'ABC']].dot(w)
#df = df.pct_change()
#df.dropna(inplace=True)
#print (df.head())
#
##returns = generate_returns(df, 10000)
#var = var_discrete(df, 0.95, 'P')
#es = es_discrete(df, 0.95, 'P')
#print ("VaR: {:.4f}".format(var))
#print ("ES: {:.4f}".format(es))
#


#from datetime import date
#from dateutil.relativedelta import relativedelta
#from finmarkets import CreditCurve, call
#from scipy.stats import norm
#import numpy as np
#import time
#dt = 1/365
#K = 110
#sigma = 0.15
#r = 0.03
#T = 3
#R = 0.4
#Q = [0.9, 0.8, 0.7]
#S0 = 105
#obs_date = date.today()
#pillars = [obs_date + relativedelta(years=i) for i in range(1, T+1)]
#cc = CreditCurve(obs_date, pillars, Q)
#t1 = time.time()
#scenarios = 500
#cvas = []
#for s in range(scenarios):
#    St = S0
#    cva = 0
#    for t in range(1, 365*T):
#        St = St * np.exp((r - 0.5 * sigma**2) * dt + \
#                         sigma * np.sqrt(dt) * norm.rvs(size=1))
#        cva += call(St, K, r, sigma, "{}y".format(T - t/365)) * \
#            (cc.ndp(obs_date+relativedelta(days=t)) - \
#             cc.ndp(obs_date+relativedelta(days=t+1)))
#    cvas.append(cva*(1 - R))
#print (np.mean(cvas))
#print (time.time() - t1)

from datetime import date
from dateutil.relativedelta import relativedelta
from finmarkets import CreditCurve, call
from scipy.stats import norm
import numpy as np
import time

dt = 1/365
K = 110
sigma = 0.15
r = 0.03
T = 3
R = 0.4
Q = [1, 0.9, 0.8, 0.7]
S0 = 105
obs_date = date.today()
dates = [obs_date + relativedelta(years=i) for i in range(T+1)]
cc = CreditCurve(obs_date, dates, Q)
t1 = time.time()
scenarios = 50
St = np.zeros(shape=(365*T, scenarios))
St[0, :] = S0
ndps = np.zeros(shape=(365*T,))
for i in range(1, 365*T):
    St[i, :] = St[i-1, :] * np.exp((r - 0.5 * sigma**2) * dt \
        + sigma* np.sqrt(dt) * norm.rvs(size=scenarios))
    ndps[i] = cc.ndp(obs_date+relativedelta(days=i)) - \
        cc.ndp(obs_date+relativedelta(days=i+1))

ts = np.array(["{}y".format(t) for t in np.arange(T-1/365, 0, -1/365)])
cvas = np.zeros(shape=(365*T, scenarios))
for s in range(scenarios):
    for j in range(len(ts)):
        cvas[j+1, s] = call(St[:365*T-1, s], K, r, sigma, ts[j])*(1-R)*ndps[1:]
cvas = np.sum(cvas, axis=0)
print (np.mean(cvas))
print (time.time() - t1)




import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from finmarkets import CreditCurve, call
from scipy.stats import norm
dt = 1/365
K = 110
sigma = 0.50
r = 0.03
T = 1
R = 0.4
Q = [1, 0.7]
S0 = 100
obs_date = date.today()
pillars = [obs_date + relativedelta(years=i) for i in range(T+1)]
cc = CreditCurve(pillars, Q)
scenarios = 10000
St = np.zeros(shape=(365*T, scenarios))
St[0, :] = S0
ndps = np.zeros(shape=(365*T,))
for i in range(1, 365*T):
    St[i, :] = St[i-1, :] * np.exp((r - 0.5 * sigma**2) * dt \
                                   + sigma* np.sqrt(dt) * norm.rvs(size=scenarios))
    ndps[i] = cc.ndp(obs_date+relativedelta(days=i)) - \
        cc.ndp(obs_date+relativedelta(days=i+1))

ts = np.arange(T-1/365, 0, -1/365)
EE = np.zeros(shape=(365*T, scenarios))
for s in range(scenarios):
    EE[1:, s] = call(St[:365*T-1, s], K, r, sigma, ts)*(1-R)*ndps[1:]
EE = np.sum(EE, axis=0)
print ("Credit VaR: {:.3f}".format(np.percentile(EE, 99.9)))


