import numpy as np

from short_rate import VasicekModel, CIRModel

print ("\nTest Short Rates")
print ("------------------")

r0 = 0.03
v = VasicekModel(0.3, 0.10, 0.03)
n = 1000
T = 1
m = 100
dt = T/m
res = []

for i in range(n):
    v.setSeed(i)
    r = v.r_generator(r0, T, m)
    I = np.sum(r[1:])*dt
    res.append(np.exp(-I))
    
print ("Exact Vasicek Price: {:.4f}".format(v.ZCB(T, r0)))
print ("MC Price: {:.4f}".format(np.mean(res)))
print ("MC Std Error: {:.4f}".format(np.std(res)/np.sqrt(n)))


c = CIRModel(0.3, 0.07, 0.03)
r = c.r_generator(0.01875, 10)
print (r)
