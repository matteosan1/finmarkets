import numpy as np
from scipy.optimize import minimize
import pandas as pd
from datetime import date

from finmarkets import OvernightIndexSwap, generate_dates, DiscountCurve

S_6m = 2*(102.5/99.5 - 1)
print (S_6m)

coeff = [1, 4, -0.276428]
print (np.roots(coeff))

def of(x):
    return 20000*x[0] + 25000*x[1]

def cons1(x):
    return 400*x[0] + 300*x[1] - 25000
def cons2(x):
    return 300*x[0] + 400*x[1] - 27000

def cons3(x):
    return 200*x[0] + 500*x[1] - 30000

cons = [{"type":"ineq", "fun":cons1},
        {"type":"ineq", "fun":cons2},
        {"type":"ineq", "fun":cons3}]

x0 = [10, 10]
bounds = [(0, 100) for _ in range(len(x0))]
r = minimize(of, x0, bounds=bounds, constraints=cons)
print (r)

obs_date = date.today()
df = pd.read_csv('https://raw.githubusercontent.com/matteosan1/finance_course/develop/input_files/ois_quotes.csv')
swaps = []

for i in range(len(df)):
    swap = OvernightIndexSwap(1e6,
                              obs_date,
                              "{:.0f}m".format(df.iloc[i]['maturity']),
                              df.iloc[i]['rate'])
    swaps.append(swap)

obs_date = date.today()
pillar_dates = []
for swap in swaps:
    pillar_dates.append(swap.payment_dates[-1])
    pillar_dates = sorted(pillar_dates)

def objective_function(x, obs_date, pillar_dates):
    curve = DiscountCurve(obs_date, pillar_dates, x)
    sum_sq = 0.0
    for swap in swaps:
        sum_sq += swap.npv(curve)**2
    return sum_sq

x0 = [1.0 for i in range(len(pillar_dates))]
bounds = [(0.01, 10.0) for i in range(len(pillar_dates))]
bounds[0] = (1.0, 1.0)

result = minimize(objective_function, x0, bounds=bounds,
                  args=(obs_date, pillar_dates))
print (result)

C = np.array([[102, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [2.5, 102.5, 0, 0, 0, 0, 0, 0, 0, 0],
              [3, 3, 103, 0, 0, 0, 0, 0, 0, 0],
              [3.5, 3.5, 3.5, 103.5, 0, 0, 0, 0, 0, 0],
              [4, 4, 4, 4, 104, 0, 0, 0, 0, 0],
              [4.5, 4.5, 4.5, 4.5, 4.5, 104.5, 0, 0, 0, 0],
              [5, 5, 5, 5, 5, 5, 105, 0, 0, 0],
              [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 105.5, 0, 0],
              [6, 6, 6, 6, 6, 6, 6, 6, 106, 0],
              [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 106.5]])
P = np.array([96.60, 93.71, 91.56, 90.24, 89.74,
              90.04, 91.09, 92.82, 95.19, 98.14])
Cinv = np.linalg.pinv(C)
d = Cinv.dot(P.T)
print (d)

for i in range(10):
    print ("yield y{}: {:.3f}%".format(i+1, -np.log(d[i])/(i+1)*100))
           
