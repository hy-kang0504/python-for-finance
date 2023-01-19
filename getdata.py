import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import streamlit as st
import statistics
import json
import pandas_ta as ta
from pandas_ta import momentum
import datetime
from datetime import datetime,timedelta
import yfinance as yf
import cryptocompare as cc
import scipy.stats as stats
from scipy.stats import norm
import re

today = datetime.today().strftime('%Y-%m-%d')

stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'BRK-A','TSLA','UNH','JNJ','XOM','V']

def stock_price_data(stocks, startdate, enddate):
    df = yf.download(stocks, start=startdate, end=enddate)['Adj Close']
    df.index = pd.to_datetime(df.index, format="%Y%m%d").to_period('D')
    return df

def stock_returns_data(stocks, startdate, enddate):
    df = yf.download(stocks, start=startdate, end=enddate)['Adj Close']
    df = df.pct_change().dropna()
    df.index = pd.to_datetime(df.index, format="%Y%m%d").to_period('D')
    return df

tokens = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'MATIC-USD', 'DOT-USD', 'TRX-USD', 'SHIB-USD', 'UNI7083-USD', 'AVAX-USD', 'LEO-USD', 'LTC-USD', 'ATOM-USD', 'LINK-USD', 'ETC-USD', 'FTT-USD', 'XLM-USD']

def token_price_data(tokens, startdate, enddate):
    df = yf.download(tokens, start=startdate, end=enddate)['Adj Close']
    df.index = pd.to_datetime(df.index, format="%Y%m%d").to_period('D')
    df = df.rename(columns=lambda x: re.sub('-USD','',x))
    return df

def token_returns_data(tokens, startdate, enddate):
    df = yf.download(tokens, start=startdate, end=enddate)['Adj Close']
    df = df.pct_change()
    df.index = pd.to_datetime(df.index, format="%Y%m%d").to_period('D')
    df = df.rename(columns=lambda x: re.sub('-USD','',x))
    return df

def token_log_returns(tokens, startdate, enddate):
    df = yf.download(tokens, start=startdate, end=enddate)['Adj Close']
    df = np.log(df.pct_change() + 1)
    df.index = pd.to_datetime(df.index, format="%Y%m%d").to_period('D')
    df = df.rename(columns=lambda x: re.sub('-USD','',x))
    return df

#     commodities, gld,  REIT,  TIPS,  LT T, HYbond,US Bnd,US E, devE,  emE, 
indices = ['DBC', 'GLD', 'VNQ', 'TIP', 'TLT', 'HYG','BND','VTI', 'VEA', 'VWO','SPY', 'QQQ']

def index_price_data(indices, startdate, enddate):
    df = yf.download(indices, start=startdate, end=enddate)['Adj Close']
    df.index = pd.to_datetime(df.index, format="%Y%m%d").to_period('D')
    return df

def index_returns_data(indicies, startdate, enddate):
    df = yf.download(indicies, start=startdate, end=enddate)['Adj Close']
    df = df.pct_change().dropna()
    df.index = pd.to_datetime(df.index, format="%Y%m%d").to_period('D')
    return df


#if you are using daily returns - use 252 or 365
#if you are using monthly returns - use 12

def annualized_returns(df, periods_per_year):
    compounded_growth = (1+df).prod()
    n_periods = df.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualized_std(df, periods_per_year):
    return df.std()*np.sqrt(periods_per_year)

def annualized_sharpe_ratio(df, rf, periods_per_year):
    rf_per_period = 1/periods_per_year
    excess_return = df - rf_per_period
    annualized_excess_return = annualized_returns(excess_return, periods_per_year)
    annualized_volatility = annualized_std(df, periods_per_year)
    return annualized_excess_return/annualized_volatility

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def normdist_test(r, alpha=0.01):
    if isinstance(r, pd.Series):
        statistic, p_value = stats.jarque_bera(r)
        return p_value > alpha       
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(normdist_test)


    
#DOWNSIDE RISK MEASURES
def drawdown2(r):

    wealth_index = 1000*(1+r).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    if isinstance(r, pd.Series):
        return pd.DataFrame({"Wealth": wealth_index, 
                             "Previous Peak": previous_peaks, 
                             "Drawdown": drawdowns})
    elif isinstance(r, pd.DataFrame):
        L = [wealth_index, previous_peaks, drawdowns]
        df = (pd.concat(L, 
               axis=1, 
               keys=('Wealth Index', 'Previous Peaks', 'Drawdowns'))
        .swaplevel(1, 0, axis=1)
        .sort_index(axis=1))
        return df
    
def drawdown(r):

    wealth_index = 1000*(1+r).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    if isinstance(r, pd.Series):
        return pd.DataFrame({"Wealth": wealth_index, 
                             "Previous Peak": previous_peaks, 
                             "Drawdown": drawdowns})
    elif isinstance(r, pd.DataFrame):
        L = [wealth_index, previous_peaks, drawdowns]
        df = (pd.concat(L, 
               axis=1, 
               keys=('Wealth Index', 'Previous Peaks', 'Drawdowns'))
             .sort_index(axis=1))
        return df

#if alpha = 5 it means that your confidence level is 95%

def var_historic(r, alpha=5):
    
    if isinstance(r, pd.Series):
        return -np.percentile(r, alpha)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, alpha=alpha)
    

def var_normdist(r, alpha=5, modified=False):
    
    z = norm.ppf(alpha/100)
    if modified: # modify the Z score based on observed skewness and kurtosis based on Cornish-Fisher Z
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36)
    return -(r.mean() + z*r.std(ddof=0))
        
    
def cvar(r, alpha=5):
    if isinstance(r, pd.Series):
        beyond_var = r <= -var_historic(r, alpha=alpha)
        return -r[beyond_var].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar, alpha=alpha)

#use return df
def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, cov_matrix):
    return (weights.T @ cov_matrix @ weights)**0.5
        
#EFFICIENT FRONTIER 

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def weight_ew(df):
    n = len(df.columns)
    return pd.Series(1/n, index=df.columns)

def min_vol(df, periods_per_year):
    er = expected_returns.mean_historical_return(df, returns_data=True, frequency=periods_per_year)
    cov_matrix = (risk_models.CovarianceShrinkage(df, returns_data=True, frequency=periods_per_year)
                    .ledoit_wolf(shrinkage_target='constant_variance'))
    ef = EfficientFrontier(er, cov_matrix)
    weights = ef.min_volatility()
    portfolio_return = ef.portfolio_performance(verbose=False)
    return np.array(list(dict(weights).values()))

from scipy.optimize import minimize
def msr(df, rf, periods_per_year):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    er = expected_returns.mean_historical_return(df, returns_data=True, frequency=periods_per_year)
    cov_matrix = (risk_models.CovarianceShrinkage(df, returns_data=True, frequency=periods_per_year)
                    .ledoit_wolf(shrinkage_target='constant_variance'))
    n = er.shape[0]
    w0 = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n 
    
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(w, rf, er, cov_matrix):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(w, er)
        vol = portfolio_vol(w, cov_matrix)
        return -(r - rf)/vol
    
    weights = minimize(neg_sharpe, w0,
                       args=(rf, er, cov_matrix), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return np.round(weights.x,2)

def max_sharpe_ratio(df, rf, periods_per_year):    
    er = expected_returns.mean_historical_return(df, returns_data=True, frequency=periods_per_year)
    cov_matrix = (risk_models.CovarianceShrinkage(df, returns_data=True, frequency=periods_per_year)
                        .ledoit_wolf(shrinkage_target='constant_variance'))
    ef = EfficientFrontier(er, cov_matrix, weight_bounds=(0,1))
    weights = ef.max_sharpe(risk_free_rate=rf)
    weights = ef.clean_weights()
    portfolio_return = ef.portfolio_performance(verbose=False)
    return np.array(list(dict(weights).values()))

import riskfolio as rp
def risk_parity(df, rf):
    port = rp.Portfolio(returns=df)
    port.assets_stats(method_mu='hist', method_cov='ledoit', method_kurt=None, d=0.94)
    w = port.rp_optimization(model='Classic', rm='MV', rf=rf, b=None, hist=True).values.ravel()
    return w

def target_return(df, target_return, market_neutral=False, periods_per_year=365):    
    try:
        er = expected_returns.mean_historical_return(df, returns_data=True, frequency=periods_per_year)
        cov_matrix = (risk_models.CovarianceShrinkage(df, returns_data=True, frequency=periods_per_year)
                        .ledoit_wolf(shrinkage_target='constant_variance'))
        ef = EfficientFrontier(er, cov_matrix, weight_bounds=(0,1))
        weights = ef.efficient_return(target_return)
        return np.array(list(dict(weights).values()))
    except ValueError:
        print('target_return must be lower than the maximum possible return')
        
def optimal_weights(df, periods_per_year):
    er = expected_returns.mean_historical_return(df, returns_data=True, frequency=periods_per_year)
    cov_matrix = (risk_models.CovarianceShrinkage(df, returns_data=True, frequency=periods_per_year)
                        .ledoit_wolf(shrinkage_target='constant_variance'))
    target_rs = np.linspace(er.min(), er.max(), 50)
    
    weights = []
    for tr in target_rs:
        weights.append(target_return(df,tr))
        
    weights = [x for x in weights if x is not None]    
    return weights     
        
def plot_ef(df, risk_free_rate=0, style='.-', legend=False, periods_per_year=365):
    """
    Plots the multi-asset efficient frontier
    """
    er = expected_returns.mean_historical_return(df, returns_data=True, frequency=periods_per_year)
    cov_matrix = (risk_models.CovarianceShrinkage(df, returns_data=True, frequency=periods_per_year)
                        .ledoit_wolf(shrinkage_target='constant_variance'))
    weights = optimal_weights(df, periods_per_year)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov_matrix) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend, color = 'blue', figsize=(12,6))
    
    #if show_cml:
    ax.set_xlim(left = 0)
    # get MSR
    w_msr = max_sharpe_ratio(df, risk_free_rate, periods_per_year)
    r_msr = portfolio_return(w_msr, er)
    vol_msr = portfolio_vol(w_msr, cov_matrix)
    cml_x = [0, vol_msr]
    cml_y = [risk_free_rate, r_msr]
    ax.plot(cml_x, cml_y, color='green', label = 'cml', linestyle='dashed', linewidth=2, markersize=10)
    ax.plot([vol_msr],[r_msr], color ='red', marker='*', label = 'Optimal Portfolio', markersize=10)
    #if show_ew: #equally weighted
    n = er.shape[0]
    w_ew = np.repeat(1/n, n)
    r_ew = portfolio_return(w_ew, er)
    vol_ew = portfolio_vol(w_ew, cov_matrix)
    ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10, label = 'equally weighted')
    #if show_gmv: #global minimum variance
    w_gmv = min_vol(df, periods_per_year)
    r_gmv = portfolio_return(w_gmv, er)
    vol_gmv = portfolio_vol(w_gmv, cov_matrix)
    ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10, label= 'minimum variance')
    
    plt.xlim(0, 1.5)
    return ax

def backtest_ws(df, estimation_window=365, weighting=weight_ew, verbose=False, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = len(df)
    # start starts at 0
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window)]
    weights = [weighting(df.iloc[window[0]:window[1]], **kwargs) for window in windows]
    # convert List of weights to DataFrame
    weights = pd.DataFrame(weights, index=df.iloc[estimation_window:].index, columns=df.columns)
    returns = (weights * df).sum(axis="columns",  min_count=1).dropna() #mincount is to generate NAs if all inputs are NAs
    return weights

def summary_stats(df, rf=0.02, periods_per_year=365):
    ann_r = df.aggregate(annualized_returns, periods_per_year=periods_per_year)
    ann_vol = df.aggregate(annualized_std, periods_per_year=periods_per_year)
    ann_sr = (ann_r - rf)/ann_vol
    dd = df.aggregate(lambda df: drawdown(df).Drawdown.min())
    norm_var5 = df.aggregate(var_normdist, modified=False)
    cf_var5 = df.aggregate(var_normdist, modified=True)
    hist_var5 = df.aggregate(var_historic)
    hist_cvar5 = df.aggregate(cvar)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": ann_sr,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic VaR (5%)": hist_var5,
        "Gaussian VaR (5%)": norm_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Max Drawdown": dd
    })


#CPPI Constant Proportion Portfolio Insurance
#use pct_change() data for this! 

def get_cppi(risky_r, safe_r=None, periods_per_year = 365, m=3, initial_value=1000, floor=0.8, rf=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    #시작할때 셋업
    # set up the CPPI parameters
    #account_value is the CPPI
    
    n_steps = len(risky_r)
    account_value = initial_value
    floor_value = initial_value*floor
    previous_peak = account_value
    
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["Risky Assets"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = rf/periods_per_year # if it's a daily df, use 365 or 252, if it's a monthly df use 12
        
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floor_val_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for i in range(n_steps):
        if drawdown is not None:
            previous_peak = np.maximum(previous_peak, account_value)
            floor_value = previous_peak*(1-drawdown)
            
        cushion = (account_value - floor_value)/account_value #cushion is a %
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        
        # update account_value at the end of every step
        account_value = risky_alloc*(1+risky_r.iloc[i]) + safe_alloc*(1+safe_r.iloc[i])
        
        # save history
        cushion_history.iloc[i] = cushion
        risky_w_history.iloc[i] = risky_w
        account_history.iloc[i] = account_value
        floor_val_history.iloc[i] = floor_value
        peak_history.iloc[i] = previous_peak
    
    #risky_wealth is a df that has the same len as risky_r
    risky_wealth = initial_value*(1+risky_r).cumprod()
    
    result = {
        "CPPI": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": initial_value,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "previous_peak": peak_history,
        "floor": floor_val_history
    }
    
    return result

import ipywidgets as widgets
from IPython.display import display

#Monte Carlo Simulation

def gbm(n_years = 5, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=4, s0=100, prices=True):
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    r_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    r_plus_1[0] = 1
    ret_val = s0*pd.DataFrame(r_plus_1).cumprod() if prices else r_plus_1 - 1
    return ret_val


from ipywidgets import interact, interactive, fixed, interact_manual

def plot_gbm(n_scenarios, mu, sigma, s0):
    wealth = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s0=s0)
    terminal_wealth = wealth.iloc[-1]
    
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, color="indianred", alpha = 0.5, linewidth=0.8)
    wealth_ax.axhline(y=s0, ls=":", color="black")
    wealth_ax.plot(0,s0, marker='o',color='darkred', alpha=0.5)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=s0, ls=':', color='black')
    
    a, b = stats.t.interval(alpha=0.95, df=len(terminal_wealth)-1, loc=np.mean(terminal_wealth), scale=stats.sem(terminal_wealth))
    print('Using a 95% confidence level, the price in the next 5years will be in between {:.0f} and {:.0f}'.format(a, b))
    

#CPPI monte carlo : generate simulated returns of risky_r to be used to calculate CPPI
def plot_cppi(n_scenarios=50, periods_per_year=365, mu=0.07, sigma=0.15, m=3, floor=0., rf=0.03, y_max=100):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    initial_value = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = get_cppi(risky_r=pd.DataFrame(risky_r),rf=rf,m=m, initial_value=initial_value, floor=floor)
    wealth = btr["CPPI"]
    # calculate terminal wealth stats
    y_max=wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    # Plot!
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=initial_value, ls=":", color="black")
    wealth_ax.axhline(y=initial_value*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=initial_value, ls=":", color="black")    
    

import cvxpy as cp    
def tracking_error(weights, portfolio_returns, benchmark_returns):
    xi= (weights @ portfolio_returns.T - benchmark_returns)
    mean = cp.sum(xi)/len(benchmark_returns)
    return cp.sum_squares(xi - mean)


def min_te(portfolio_prices, portfolio_returns, benchmark_returns):
    er = expected_returns.mean_historical_return(portfolio_prices)
    cov_matrix = risk_models.sample_cov(portfolio_prices)
    ef = EfficientFrontier(er, cov_matrix)
    ef.add_constraint(lambda weights: tracking_error(weights, portfolio_returns, benchmark_returns) <= 0.1**2)
    ef.min_volatility()
    weights = ef.clean_weights()
    return np.array(list(dict(weights).values()))


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def linear_regression(df, exp_var, dep_var):
    n = len(exp_var)
    summ = pd.DataFrame(index = exp_var, columns=dep_var)    
    X = df[exp_var]
    y = df[dep_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    for i in dep_var:
        lm=LinearRegression()
        lm.fit(X_train, y_train[i])
        summ[i] = lm.coef_[:n]
    
    summ=summ.T
    summ['Intercept']=lm.intercept_        
    return summ

def get_macro_indicators():
    return pd.read_csv('data/current_monthly.csv').iloc[0,1:]

import api_config
from fredapi import Fred
fred = Fred(api_key = api_config.fred_api_key)

def fred_data():
    
    indicators = get_macro_indicators()    
    indicator_names = indicators.index.to_list()

    df = pd.DataFrame()

    for i in indicator_names:
        try:
            s = fred.get_series(i)
            s.name = i
            df = df.join(s, how='outer')
            df.index.name='Date'
        except Exception:
            pass
        
    df = df.dropna(thresh=80)

    for i in df.columns:
        code = indicators[i]
        if code == 1:
            df[i].apply(lambda x: x)
        elif code == 2:
            df[i] = df[i].diff()
        elif code == 3:
            df[i] = df[i].diff(periods=2)
        elif code == 4:
            df[i] = df[i].apply(np.log)
        elif code == 5:
            df[i] = df[i].apply(np.log)
            df[i] = df[i].diff(periods=2)
        elif code == 6:
            df[i] = df[i].apply(np.log)
            df[i] = df[i].diff(periods=2)
        elif code == 7:
            df[i] = df[i].pct_change()
            df[i] = df[i].diff()
    
    df = df.iloc[2:-1]
    df = df.fillna(0)
    
    lags = [1,3,6,12]
    newdf = pd.concat([df.shift(t).add_suffix(f"_lag_{t}M") for t in lags], axis=1)
    df = pd.concat([df, newdf], axis=1)
    df.dropna(inplace=True)
    df.index.name='Date'
    
    regime = pd.DataFrame(fred.get_series('USREC').rename('Regime'))

    df = pd.concat((df,regime), axis=1, join='inner')
    
    return df

#regime here is a dataframe that has a single column
def plot_regime(regime):
    startdates = []
    enddates = []
    prev_val = False

    for index, val in regime.iterrows():
        if not prev_val and val['Regime'] == 1:
            startdates.append(index)
            prev_val = True

        if prev_val and val['Regime'] == 0:
            enddates.append(index)
            prev_val = False

    if len(startdates) != len(enddates):
        enddates.append(regime.index[-1])

    crash_dates = list(zip(startdates, enddates))

    plt.figure(figsize=(18, 6))
    
    for (start, end) in crash_dates:
        plt.axvspan(start, end, color='gray', alpha=0.3)   