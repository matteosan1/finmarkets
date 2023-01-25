from scipy.stats import gumbel_l, norm
from matplotlib import pyplot as plt

gumbel = gumbel_l()
xg = gumbel.rvs(100000)
xu = gumbel.cdf(xg)
xn = norm.ppf(xu)
sub1 = plt.subplot(1, 3, 1)
sub1.hist(xg, 50)
sub1.grid(True)
sub1.set_xlabel(r"$x_{gumbel}$")
sub2 = plt.subplot(1, 3, 2)
sub2.hist(xu, 50)
sub2.grid(True)
sub2.set_xlabel(r"$x_{uniform}$")
sub3 = plt.subplot(1, 3, 3)
sub3.hist(xn, 50)
sub3.grid(True)
sub3.set_xlabel(r"$x_{normal}$")
plt.show()


from scipy.stats import beta, uniform
x = uniform.rvs(size=10)
b = beta(a=3, b=10)
x_b = b.ppf(x)
print (x_b)



from scipy.stats import lognorm, multivariate_normal, norm
import numpy
numpy.random.seed(1)
samples = 1000000
l1_uncorr = lognorm(0.5).rvs(size=samples)
l2_uncorr = lognorm(0.5).rvs(size=samples)
mvnorm = multivariate_normal(mean = (0, 0),
                             cov = [[1, 0.8],
                                    [0.8, 1]])

x = mvnorm.rvs(size=samples)
x_corr = norm.cdf(x)
l1_corr = lognorm(0.5).ppf(x_corr[:, 0])
l2_corr = lognorm(0.5).ppf(x_corr[:, 1])
plt.figure(figsize=(12, 4))
sub1 = plt.subplot(1, 2, 1)
sub1.hist2d(l1_uncorr, l2_uncorr, range=[[0, 2], [0, 2]], bins=(100, 100))
sub2 = plt.subplot(1, 2, 2)
sub2.hist2d(l1_corr, l2_corr, range=[[0, 2], [0, 2]], bins=(100, 100))
plt.show()

default_6m = lognorm(0.5).cdf(.5)
print(default_6m**2)

success = 0.0
for i in range(samples):
    if max(x_corr[i]) < default_6m:
        success += 1
print (success/samples)




import numpy as np
from scipy.stats import norm

PDF1 = np.array([0.065, 0.081, 0.072, 0.064, 0.059])
PDF2 = np.array([0.238, 0.152, 0.113, 0.092, 0.072])
C1 = PDF1.cumsum()
C2 = PDF2.cumsum()
for c in C1:
    print ("{:.3f} −> {:.4f}".format(c, norm.ppf(c)))
print ("")
for c in C2:
    print ("{:.3f} −> {:.4f}".format(c, norm.ppf(c)))

from scipy.stats import multivariate_normal
g = multivariate_normal(mean=[0,0],
                        cov=[[1, 0.4],
                             [0.4, 1]])
print (g.cdf([-1.5141, -0.7128]))


