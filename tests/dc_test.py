import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from finmarkets import DiscountCurve

print ("\nTest DiscountCurve")
print ("------------------")

df = pd.read_excel("https://github.com/matteosan1/finance_course/raw/master/input_files/discount_factors_2022-10-05.xlsx")

obs_date = date.today()
pillars = [obs_date + relativedelta(months=i) for i in df['months']]
dfs = df['dfs'].values
dc = DiscountCurve(obs_date, pillars, dfs)
df_date = obs_date + relativedelta(days=195)
df0 = dc.df(df_date)
print ("discount factor at {}: {:.4f}".format(df_date, df0))
