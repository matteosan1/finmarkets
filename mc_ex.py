from random import seed, randint

seed(1)
trials = 10000
success = 0
for _ in range(trials):
    d1, d2, d3 = randint(1, 6), randint(1, 6), randint(1, 6)
    if d1 == d2 and d2 == d3:
        success += 1
print ("The probability to get three equal dice is {:.4f}".format(success/trials))

import random
random.seed(1)
successes = {"=1":0.0, "=4":0.0, "<13":0.0}
trials = 1000
for _ in range(trials):
    d1 = random.randint(1, 6)
    d2 = random.randint(1, 6)
    if (d1 + d2) == 1:
        successes["=0"] += 1.0
    if (d1 + d2) == 4:
        successes["=4"] += 1.0
    if (d1 + d2) < 13:
        successes["<13"] += 1.0
for k,v in successes.items():
    print ("P({}): {:.3f}".format(k, v/trials))

import numpy as np
def mc_normal(mean, std_dev, samples):
    results = []
    for _ in range(samples):
        results.append(np.random.normal(mean, std_dev))
    return np.array(results)

s = 100000
upper_limit = 34
component_1 = mc_normal(5,1,s)
component_2 = mc_normal(10,1,s)

component_3 = mc_normal(15,1,s)
total = component_1 + component_2 + component_3
probability = np.sum(total > upper_limit)/len(total)*100
print("Probability of exceeding the time limit: ", round(probability, 3), "%")

import numpy as np
from random import randint
k = 20
sim = 10000
points = [0 for _ in range(51)]
for s in range(sim):
    for k in range(1, 51):
        p = 0
        while p < k:
            d = randint(1, 6)
            if d == 1:
                p = 0
                break
            else:
                p += d
            points[k] += p
        points = np.array(points)/sim


import random
flasks = ["C"]*54 + ["U"] * 6
random.seed(1)
trials = 1000
success = 0.
for _ in range(trials):
    draw = random.sample(flasks, 5)
    if draw.count("U") == 3:
        success += 1.
print ("Probability: {:.3f}%".format(success/float(trials)*100))

from scipy.stats import norm
import numpy as np
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/matteosan1/finance_course/master/input_files/histo_temp.csv")
temperatures = df['T']
alpha = 0.99
A = norm.ppf((1 + alpha)/2)
m, se = np.mean(temperatures), np.std(temperatures)
h = A*se/np.sqrt(len(temperatures))
print ("Avg temperature in September (US): {:.1f}".format(m))
print ("{:.1f}% confidence interval: +- {}".format(alpha*100, h))
