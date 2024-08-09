from scipy.integrate import quad
from numpy import percentile, random

def var_continuous(f, alpha=0.95):
    """
    Computes VaR at a specified confidence level, given a continuous loss distribution
    
    Params:
    -------
    f: scipy.stats.rv_continuous
        continuous distribution representing the portfolio losses
    alpha: float
        confidence level for VaR calculation
    """
    return -f.ppf(1-alpha)

def es_continuous(f, alpha=0.95):
    """
    Computes Expected Shortfall at a specified confidence level, given a continuous loss distribution
    
    Params:
    -------
    f: scipy.stats.rv_continuous
        continuous distribution representing the portfolio losses
    alpha: float
        confidence level for ES calculation
    """
    def integrand(x, f):
        return f.ppf(x)
    alpha = 1-alpha
    I = quad(integrand, 0, alpha, args=(f,))
    return -1/alpha*I[0]
  
def generate_returns(df, N=10000, seed=1):
    random.seed(seed)
    data = df.reset_index()
    return data.loc[random.choice(range(len(data)), N)]

def var_discrete(df, alpha=0.95, return_col="P", N=10000):
    """
    Computes VaR at a specified confidence level, given a discrete loss distribution as a DataFrame
    
    Params:
    -------
    df: pandas.DataFrame
        dataset containing the loss descrete distribution
    alpha: float
        confidence level for VaR calculation
    return_col: str
        name of the column in the dataframe
    N: int
        number of samples to generate
    """
    alpha = 1-alpha
    new_df = generate_returns(df, N)
    print (new_df.head())
    return -percentile(new_df[return_col], alpha*100)
    
def es_discrete(df, alpha=0.95, return_col="P", N=10000):
    """
    Computes Expected Shortfall at a specified confidence level, given a discrete loss distribution
    
    Params:
    -------
    df: pandas.DataFrame
        dataset containing the loss descrete distribution
    alpha: float
        confidence level for ES calculation
    return_col: str
        name of the column in the dataframe
    N: int
        number of samples to generate
    """
    alpha = 1-alpha
    new_df = generate_returns(df, N)
    var = percentile(new_df[return_col], alpha*100)	
    return -df[df<=var][return_col].mean()
