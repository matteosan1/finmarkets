import pandas as pd

df = pd.read_csv("https://github.com/matteosan1/finance_course/raw/master/input_files/share_price.csv", index_col='Date')
daily_returns = df.pct_change()
returns = daily_returns.mean()*252
covariance = daily_returns.cov()*252
print (returns)
print (covariance)





import numpy as np
from scipy.optimize import minimize

def sum_weights(w):
    return np.sum(w) - 1

def utility(w, returns, cov, risk_aversion):
    return -(returns.dot(w) - 0.5*w.T.dot(cov.dot(w))*risk_aversion)

num_assets = 10
constraints = [{'type': 'eq', 'fun': sum_weights},]
bounds = tuple((0, 1) for _ in range(num_assets))
weights = [1./num_assets for _ in range(num_assets)]
for risk_aversion in (1, 4, 10):
    opts = minimize(utility, weights, args=(returns, covariance, risk_aversion),
                    bounds=bounds, constraints=constraints)
    
for i, c in enumerate(df.columns):
    print ("{} {:.1f}".format(c, opts.x[i]*100), end=" ")
print()
print ("Expected return: {:.3f}".format(opts.x.dot(returns)))
print ("Variance: {:.3f}".format(opts.x.T.dot(covariance.dot(opts.x))))





import pandas as pd

df = pd.read_csv("https://github.com/matteosan1/finance_course/raw/master/input_files/share_price.csv", index_col='Date')
daily_returns = df.pct_change()
returns = daily_returns.mean()*252
covariance = daily_returns.cov()*252




import numpy as np
from scipy.optimize import minimize

def sum_weights(w):
    return np.sum(w) - 1

def min_risk(w, cov):
    return np.sqrt(w.T.dot(cov.dot(w)))

num_assets = 10
constraints = ({'type': 'eq', 'fun': sum_weights},)
bounds = tuple((0, 1) for asset in range(num_assets))
weights = [1./num_assets for _ in range(num_assets)]
opts = minimize(min_risk, weights, args=(covariance,),
                bounds=bounds, constraints=constraints)
print (opts)

print ("Portfolio composition")
for i, n in enumerate(df.columns):
    print ("{:5}: {:4.1f}%".format(n, opts.x[i]*100))
print ("Portfolio variance: {:.4f}".format(opts.fun**2))
print ("Expected Portfolio return: {:.3f}".format(opts.x.dot(returns)))






def risk_parity(w, cov):
    variance = w.T.dot(cov.dot(w))    
    sum = 0
    N = len(w)
    for i in range(N):
        sum += (w[i] - (variance/(N*(cov.dot(w))[i])))**2
    return sum

args = (covariance,)
constraints = ({'type': 'eq', 'fun': sum_weights})
bounds = tuple((0, 1) for asset in range(num_assets))
weights = [1./num_assets for _ in range(num_assets)]
opts = minimize(risk_parity, weights, args=(covariance,),
                bounds=bounds, constraints=constraints)
print (opts)

def risk_parity(w, cov):
    variance = w.T.dot(cov.dot(w))
    sum = 0
    N = len(w)
    for i in range(N):
        sum += (w[i] - (variance/(N*(cov.dot(w))[i])))**2
    return sum

args = (covariance,)
constraints = ({'type': 'eq', 'fun': sum_weights})
bounds = tuple((0, 1) for asset in range(num_assets))
weights = [1./num_assets for _ in range(num_assets)]
opts = minimize(risk_parity, weights, args=(covariance,),
                bounds=bounds, constraints=constraints)
print (opts)












