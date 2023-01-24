import numpy as np

from scipy.optimize import brentq

N = 1000
maturity = 10
coupon = 0.09
tenor = 0.5
rates = {i:0.08 for i in range(1, int((maturity/tenor)+1))}
price = 0
for tau in range(1, int((maturity/tenor)+1)):
    price += N * coupon*tenor / (1+rates[tau]*tenor)**(tau)
price += N / (1+rates[tau]*tenor)**(tau)
print ("{:.2f}".format(price))

rates = {i:0.09 for i in range(1, int((maturity/tenor)+1))}
price = 0
for tau in range(1, int((maturity/tenor)+1)):
    price += N * coupon*tenor / (1+rates[tau]*tenor)**(tau)
price += N / (1+rates[tau]*tenor)**(tau)
print ("{:.2f}".format(price))

#########################

def ytom(y, N, C, P0, maturity_years, tenor=1):
    price = 0
    for p in range(1, int((maturity_years/tenor)+1)):
        price += C*tenor*N/(1+y*tenor)**p
    price += N/(1+y*tenor)**p - P0
    return price

print (brentq(ytom, -0.3, 1, args=(100, 0.10, 100.917, 3, 1)))

#########################

def zero_bond_pv(N, r, maturity):
    return N / (1+r)**(maturity)

def zero_mac_duration(P0, maturity):
    return maturity * P0/P0

def bond_pv(N, C, r, maturity, tenor=1):
    price = 0
    for tau in range(1, int((maturity/tenor)+1)):
        price += N * C*tenor / (1+r*tenor)**(tau)
    price += N / (1+r*tenor)**(tau)
    return price

def mac_duration(N, C, y, maturity, tenor=1):
    P0 = bond_pv(N, C, y, maturity, tenor)
    d=0
    for tau in range(1, int((maturity/tenor)+1)):
        d += tau*N*C*tenor/(1+y*tenor)**tau/P0
    d += tau*N/(1+y*tenor)**tau/P0
    return P0, d

P0, dur = mac_duration(1000, 0.12, 0.12, 7)
print ("Bond price: {:.2f}".format(P0))
print ("Duration: {:.2f}".format(dur))
PZ = zero_bond_pv(1120, 0.12, 7)
print ("Zero Bond price: {:.2f}".format(PZ))
durZ = zero_mac_duration(PZ, 7)
print ("Zero Duration: {:.2f}".format(durZ))

P1 = bond_pv(1000, 0.12, 0.13, 7)
print ("Bond price: {:.2f}".format(P1))
print ("Capital loss Bond: {:.3f}%".format(100*(P1-P0)/P0))
PZ1 = zero_bond_pv(1120, 0.13, 7)
print ("Bond price: {:.2f}".format(PZ1))
print ("Capital loss Zero Bond: {:.3f}%".format(100*(PZ1-PZ)/PZ))

closs1 = -dur*0.01
print ("Capital loss1: {:.3f}%".format(closs1*100))
closs2 = -durZ*0.01
print ("Capital loss1: {:.3f}%".format(closs2*100))

