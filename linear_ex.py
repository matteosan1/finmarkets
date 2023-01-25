import pandas as pd
import statsmodels.api as sm

closing = pd.read_csv("https://github.com/matteosan1/finance_course/raw/master/input_files/dji.csv", index_col='Date')
DJI = closing['DJI'].copy()
closing.drop(columns='DJI', inplace=True)

DJI.index = DJI.index.strftime('%Y-%m-%d')
model = sm.OLS(DJI[:'2020-07-31'], closing[:'2020-07-31']).fit()
print (model.summary())

selected_columns = list(model.pvalues[model.pvalues<0.05].index)
model_small = sm.OLS(DJI[:'2020-07-31'],
closing[selected_columns][:'2020-07-31']).fit()
print (model_small.summary())

residuals = DJI-model_small.predict(closing[selected_columns])







P = 4000 + 6000 + 12000 + 3000
rp =(4000/P)*0.18 + (6000/P)*0.08 + (12000/P)*0.16 +(3000/P)*0.12
print ("Portfolio Return: {:.2f}%".format(rp*100))






rm = 0.115
rf = 0.04
beta_p = 0.35 * 1 + 0.35 * 3 + 0.3 * 0.5
rp = rf + beta_p*(rm - rf)
print ("Portfolio beta: {:.2f}".format(beta_p))
print ("Portfolio return: {:.3f}%".format(rp*100))

Er_msft = rf + 1*(rm-rf)
Er_amzn = rf + 3*(rm-rf)
Er_ge = rf + 0.5*(rm-rf)
rp2 = Er_msft * 0.35 + Er_amzn *0.35 + Er_ge * 0.3

print("Er (MSFT): {:.2f}%".format(Er_msft*100))
print("Er (AMZN): {:.2f}%".format(Er_amzn*100))
print("Er (GE): {:.2f}%".format(Er_ge*100))
print("Portfolio return: {:.3f}%".format(rp2*100))


