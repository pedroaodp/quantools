import numpy as np
import pandas as pd
from  scipy.stats import norm, t
import yfinance as yf



def log_returns(prices):
    """
    compute a stock's  log return
    
    :param prices: Prices time series
    :return: Log return
    """
    log_return = np.log(prices) - np.log(prices.shift(1))
    return log_return


def VaR(prices, ci, approach= 'p'):
    """
    compute a stock's  Value at Risk using parametric or historical approach.
    
    :param prices: Prices time series
    :param ci: Confidence Interval
    :return: Historical o Parametric VaR
    """
    if ci > 1:
        ci = ci/100
    elif 0 < ci < 1:
        ci = ci

    returns = log_returns(prices)
    mu = np.mean(returns)
    sigma = np.std(returns)

    sreturns = returns.sort_values()
    alpha = 1- ci

    if approach == 'p':
        VaR = norm.ppf(alpha, mu, sigma)
    elif approach == 'h':
        VaR = sreturns.quantile(1 - ci)
    else:
        raise TypeError('The approach should be parametric or historical')

    return VaR



def CVaR(prices, ci):
    """
    This function return CVaR for a normal distribution 
    
    :param prices: Prices time serie
    :param ci: Confidence Interval
    :return: VaR and CVaR of a t-distribution
    """
    if ci > 1:
        ci = ci / 100
    elif 0 < ci < 1:
        ci = ci

    returns = log_returns(prices)
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = 1- ci
    CVaR = (alpha)**(-1) *norm.pdf(norm.ppf(alpha))*sigma-mu


    return CVaR

def tVaR(prices, ci, dof=0):
    """
    This function return VaR and CVaR for a t-student distribution 
    
        :param prices: Prices time series
        :param ci: Confidence Interval
        :param dof: Degrees of freedom
        :return: VaR and CVaR of a t-distribution
    """
    if ci > 1:
        ci = ci / 100
    elif 0 < ci < 1:
        ci = ci

    returns = log_returns(prices)
    mu = np.mean(returns)
    sigma = np.std(returns)
    alpha = 1 - ci
    nu = dof
    x_anu = t.ppf(alpha, nu)

    VaR = np.sqrt((nu - 2) / nu) * t.ppf(1 - alpha,nu) * sigma - mu
    CVaR = (-1 / alpha) * (1 - nu) ** (-1) * (nu - 2 + x_anu ** 2) * t.pdf(x_anu,nu) * sigma - mu

    VaR = {"tVaR": VaR,
           "CVaR": CVaR}

    return VaR
