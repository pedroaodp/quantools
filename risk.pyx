import numpy as np
import pandas as pd
from  scipy.stats import norm, t
import yfinance as yf



def log_returns(prices):
    """
    compute log returns for each ticker.

    parameters
     ----------
    prices : dataframe or pandas séries
    prices for each ticker and date

     returns
     -------
    log_returns : dataframe
    log returns for each ticker and date
     """
    log_return = np.log(prices) - np.log(prices.shift(1))
    return log_return


def VaR(prices, ci, approach= 'p'):
    """
    compute the stock's  Value at Risk using parametric or historical approach

    parameters
     ----------
    prices : dataframe or pandas séries
    prices for each ticker and date

    ci: confidence interval
    90%, 95%, and 99% confidence intervals

    distribution: Normal or t-student approach
    ditribution = 'p' : Parametric approach
    distribution = 't': Historical approach

     returns
     -------
    VaR : Scalar
    Return the VaR number
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
        compute the stock's  Value at Risk using parametric approach

        parameters
         ----------
        prices : dataframe or pandas séries
        prices for each ticker and date

        ci: confidence interval
        90%, 95%, and 99% confidence intervals

        distribution: Normal or t-student approach
        ditribution = 'n' : Normal approach
        distribution = 't': t-student approach

         returns
         -------
        VaR : Scalar
        Return the VaR number
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
