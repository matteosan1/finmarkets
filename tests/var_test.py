import pandas as pd
import numpy as np

from scipy.stats import norm

from finmarkets import var_continuous, es_continuous, var_discrete, es_discrete

print ("\nTest Risk Measures")
print ("------------------")

df = pd.read_csv("https://raw.githubusercontent.com/matteosan1/finance_course/master/input_files/historical_data.csv", index_col='Date')
df['P'] = df['aapl']*0.6 + df['nflx']*0.4
df = df.pct_change()
df.dropna(inplace=True)

mu = df.mean() 
sigma = df.std()

f = norm(mu['P'], sigma['P'])
print ("1d-95% VaR: {:.4}".format(var_continuous(f, 0.95)))
print ("1d-95% ES: {:.4}".format(es_continuous(f, 0.95)))
   
print ("1d-95% VaR (discrete): {:.4f}".format(var_discrete(df, 0.95)))
print ("1d-95% ES (discrete): {:.4f}".format(es_discrete(df, 0.95)))
