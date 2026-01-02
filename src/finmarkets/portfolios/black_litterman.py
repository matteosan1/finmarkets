
import pandas as pd

class Views:
  def __init__(self):
    self.abs = []
    self.rel = []

  def add_abs_view(self, ticker, value, interval):
    self.abs.append((ticker, value, interval))

  def add_rel_view(self, ticker1, ticker2, value, interval):
    self.abs.append((ticker1, ticker2, value, interval))

class BlackLitterman:
  def __init__(self, prices, mkt_prices, mkt_caps, riskfree_rate, freq, tau):
    self.prices = prices
    self.mkt_prices = mkt_prices
    self.mkt_caps = mkt_caps
    self.tickers = prices.columns
    self.num_assets = len(self.tickers)
    self.returns = prices.pct_change()
    self.Sigma = self.returns.cov() * freq
    self.riskfree_rate = riskfree_rate
    self.freq = freq
    self.tau = tau
    self.Q = []
    self.P = []
    self.intervals = []
    self.Omega = None
    self.delta = None
    self.Pi = None
  
  def risk_aversion(self):
    rets = self.mkt_prices.pct_change().dropna()
    r = rets.mean().values[0] * self.freq
    var = rets.var().values[0] * self.freq
    self.delta = (r - self.riskfree_rate)/var

  def equilibrium_return(self):
    mkt_weights = self.mkt_caps / self.mkt_caps.sum()
    self.Pi = self.delta * self.Sigma.values.dot(mkt_weights) + self.riskfree_rate

  def views(self, abs_views, rel_views):
    for i in range(len(abs_views)):
      self.Q.append(abs_views[i][1])
      self.intervals.append(abs_views[i][2])
      self.P.append((self.tickers == abs_views[i][0]).astype(int))
      
    for i in range(len(rel_views)):
      self.Q.append(abs_views[i][2])
      self.intervals.append(abs_views[i][3])
      p1 = (tickers == rel_views[i][0]).astype(int)*0.5
      p2 = (tickers == rel_views[i][1]).astype(int)*-0.5
      self.P.append(p1+p2)
    
    variances = []
    for lb, ub in intervals:
      sigma = (ub - lb)/2
      variances.append(sigma ** 2)

    self.Omega = np.diag(variances)
    self.Q = np.array(self.Q)
    self.P = np.array(self.P)

  def predict(self):
    tau_sigma_inv = np.linalg.inv(self.tau*self.Sigma.values)
    omega_inv = np.linalg.inv(self.Omega)
    p_omega_p = self.P.T.dot(omega_inv.dot(self.P))
    #E_R = np.linalg.inv(tau_sigma_inv + p_omega_p)@((tau_sigma_inv.dot(Pi)) + P.T.dot(omega_inv.dot(Q)))

    A = self.P.dot(self.tau*self.Sigma.values.dot(self.P.T)) + self.Omega
    b = self.Q - self.P.dot(self.Pi.flatten())
    E_R = self.Pi.flatten() + self.tau*self.Sigma.values.dot(self.P.T) @ np.linalg.solve(A, b)
    Sigma_hat = self.Sigma.values + np.linalg.inv(tau_sigma_inv + p_omega_p)

    return E_R, Sigma_hat
