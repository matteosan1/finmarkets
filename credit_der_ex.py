import numpy as np
from scipy.stats import multivariate_normal, norm

p_default = [0, 0.01, 0.02, 0.05, 0.08, 0.11]

cov = np.ones(shape=(6, 6))*0.4
np.fill_diagonal(cov, 1.0)
mvnorm = multivariate_normal(mean=[0]*6, cov = cov)

n_to_default = 4
trials = 50000
result = [0., 0., 0., 0., 0., 0.]
x = mvnorm.rvs(size=trials)
sim_defaults = np.sort(norm.cdf(x))

for s in sim_defaults:
    for i in range(1, len(p_default)):
        if p_default[i-1] <= s[n_to_default-1] <= p_default[i]:
            result[i] += 1

print ("4th−to−default probabilies")
for i in range(len(p_default)):
    print ("{}: {:.4f}".format(i, result[i]/trials))


from finmarkets import DiscountCurve, BasketDefaultSwaps
from datetime import date
from dateutil.relativedelta import relativedelta

n_cds = 10
rho = 0.15
l = 0.016
ndefaults = 5

obs_date = date.today()
start_date = obs_date
dates = [obs_date + relativedelta(years=i) for i in range(1, 4)]
dfs = [1/(1+0.03)**i for i in range(1, 4)]
dc = DiscountCurve(obs_date, dates, dfs)

basket = BasketDefaultSwaps(1e6, n_cds, l, rho, start_date, "3y", 0.01)
basket.credit_curve(obs_date, dates, ndefaults)
bkeven = basket.breakeven(dc)
print(bkeven)

new_basket = BasketDefaultSwaps(1e6, n_cds, l, rho, start_date, "3y", bkeven)
new_basket.credit_curve(obs_date, dates, ndefaults)
print(new_basket.npv(dc))



from scipy.stats import binom, norm
from scipy.integrate import quad
import numpy as np

N = 125
A = 1
R = 0
M = 1
q = 0.02
tranches = [[1,3],[4, 6],[7,125]]
def p(M, rho, lims):
    qM = max(1e-10, norm.cdf((norm.ppf(q)-np.sqrt(rho)*M)/(np.sqrt(1-rho))))
    pN = binom(N, qM)
    prob = (lims[1]-lims[0]+1) * (pN.cdf(N) - pN.cdf(lims[1]-1))
    for i in range(lims[0], lims[1]):
        prob += (i-lims[0]+1)*pN.pmf(i)
    return norm.pdf(M)*prob

res = [[],[],[]]
for i in range(len(tranches)):
    for rho in np.arange(0., 1.05, 0.1):
        if rho == 1.0:
            rho = 0.99
        v = quad(p, -np.inf, np.inf, args=(rho, tranches[i]))
        res[i].append(v[0])

print (res[0][1] + res[1][1] + res[2][1])
print (res[0][5] + res[1][5] + res[2][5])
print (res[0][9] + res[1][9] + res[2][9])



from finmarkets import DiscountCurve, CreditCurve, CollDebtObligation
from datetime import date
from dateutil.relativedelta import relativedelta

pillar_dates = []
df = []
obs_date = date.today()
start_date = obs_date
dates = [obs_date + relativedelta(years=i) for i in range(1, 2)]
dfs = [1/(1 + 0.05)**i for i in range(1, 2)]
dc = DiscountCurve(obs_date, dates, dfs)

cc = CreditCurve(obs_date,
                 [obs_date + relativedelta(years=i) for i in range(1, 5)],
                 [0.99, 0.97, 0.95, 0.93])

nnames = 125
tranches = [[0.0, 0.03], [0.03, 0.06], [0.06, 0.09], [0.09, 1.0]]
spreads = [0.15, 0.07, 0.03, 0.01]

cdo = CollDebtObligation(100e6, nnames, tranches,
                         0.3, cc, start_date, spreads,
                         "1y", "12m")

for i in range(len(tranches)):
    print ("Tranche {} ({}): {:.5f}".format(i, tranches[i], cdo.fair_value(i, dc)))



