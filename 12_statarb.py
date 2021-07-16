# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:24:24 2021

@author: Ricardo
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2


import bollinger_bands
importlib.reload(bollinger_bands)

backtest = bollinger_bands.backtest()
backtest.ric_long = 'ETH-USD' # Numerator
backtest.ric_short = 'BTC-USD' # Denominator
backtest.rolling_days = 25 # N
backtest.level_1 = 3. # a
backtest.level_2 = 4 # b
backtest.data_cut = 0.7 # 70 % in-sample and 30 % out-of-sample
backtest.data_type = 'in-sample' # in-sample out-of-sample

# Load data
backtest.load_timeseries()
data_table = backtest.data_table
cut = int(backtest.data_cut*data_table.shape[0])
if backtest.data_type == 'in-sample':
    df1 = data_table[:cut]
elif backtest.data_type == 'out-of-sample':
    df1 = data_table[cut:]
df1 = df1.reset_index(drop=True)

# Spread at current close
df1['spread'] = df1['price_x']/df1['price_y']
df1['spread'] = df1['spread']/df1['spread'][0]
df1['spread_previous'] = df1.loc[:,'spread'].copy().shift(1)
df1.loc[0, 'spread_previous'] = 1

# Compute bollinger bands
size = df1.shape[0]
columns = ['lower_2', 'lower_1', 'mean', 'upper_1', 'upper_2']
mtx_bollinger = np.zeros((size, len(columns)))
mtx_bollinger[:] = np.nan
for n in range(backtest.rolling_days+1, size):
    vec_price = df1['spread'].values
    vec_price = vec_price[n-backtest.rolling_days-1:n-1]
    mu = np.mean(vec_price)
    sigma = np.std(vec_price)
    mtx_bollinger[n][0] = mu - backtest.level_2*sigma
    mtx_bollinger[n][1] = mu - backtest.level_1*sigma
    mtx_bollinger[n][2] = mu
    mtx_bollinger[n][3] = mu + backtest.level_1*sigma
    mtx_bollinger[n][4] = mu + backtest.level_2*sigma

df2 = pd.DataFrame(data=mtx_bollinger, columns=columns)
timeseries = pd.concat([df1,df2], axis=1)
timeseries = timeseries.dropna()
timeseries = timeseries.reset_index(drop=True)

#### Plot Bollinger Bands #####
t = timeseries['date']
spread = timeseries['spread']
mu = timeseries['mean']
u1 = timeseries['upper_1']
u2 = timeseries['upper_2']
l1 = timeseries['lower_1']
l2 = timeseries['lower_2']
plt.close('all')
plt.figure()
plt.title('Spread ' + backtest.ric_long + '/' + backtest.ric_short )
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(t, mu, color='blue', label='mean')
plt.plot(t, l1, color='green', label='lower_1')
plt.plot(t, u1, color='green', label='upper_1')
plt.plot(t, l2, color='red', label='lower_2')
plt.plot(t, u2, color='red', label='upper_2')
plt.plot(t, spread, color='black', label='spread', marker='.')
plt.legend(loc=0)
plt.grid()
plt.show()


### Trading Strategy ####
size = timeseries.shape[0]
columns = ['position', 'entry_signal', 'exit_signal', 'pnl_daily', 'trade', 'pnl_trade']
position = 0
entry_spread = 0
can_trade = False
mtx_backtest = np.zeros((size, len(columns)))
for n in range(size):
    # Input data for the day
    spread = timeseries['spread'][n]
    spread_previous = timeseries['spread_previous'][n]
    upper_2 = timeseries['upper_2'][n]
    upper_1 = timeseries['upper_1'][n]
    mean = timeseries['mean'][n]
    lower_1 = timeseries['lower_1'][n]
    lower_2 = timeseries['lower_2'][n]
    # Reset output data for the day
    pnl_daily = 0
    trade = 0
    pnl_trade = 0
    # Check if we can trade
    if not can_trade:
        can_trade = position == 0 and spread > lower_1 and spread < upper_1
    if not can_trade:
        continue
    # Enter new position
    if position == 0:
        entry_signal = 0
        exit_signal = 0
        if spread > lower_2 and spread < lower_1:
            entry_signal = 1    # Buy signal
            position = 1
            entry_spread = spread
        elif spread > upper_1 and spread < upper_2:
            entry_signal = -1   # Sell signal
            position = -1
            entry_spread = spread
    # Exit long position
    elif position == 1:
        entry_signal = 0
        pnl_daily = position*(spread - spread_previous)
        if n == size - 1 or spread > mean or spread < lower_2:
            exit_signal = 1     # Last day or take profit or stop loss
            pnl_trade = position*(spread - entry_spread)
            position = 0
            trade = 1
            can_trade = False
        else:
            exit_signal = 0
    # Exit short position
    elif position == -1:
        entry_signal = 0
        pnl_daily = position*(spread - spread_previous)
        if n == size - 1 or spread < mean or spread > upper_2:
            exit_signal = 1     # Last day or take profit or stop loss
            pnl_trade = position*(spread - entry_spread)
            position = 0
            trade = 1
            can_trade = False
        else:
            exit_signal = 0
            
    # Save data for the day
    m = 0
    mtx_backtest[n][m] = position
    mtx_backtest[n][m+1] = entry_signal
    mtx_backtest[n][m+2] = exit_signal
    mtx_backtest[n][m+3] = pnl_daily
    mtx_backtest[n][m+4] = trade
    mtx_backtest[n][m+5] = pnl_trade
    
df2 = pd.DataFrame(data=mtx_backtest, columns=columns)
df = pd.concat([timeseries, df2], axis=1)
df = df.dropna()
df = df.reset_index(drop = True)
df['cum_pnl_daily'] = np.cumsum(df['pnl_daily'])

# Compute Sharpe ratio and number of trades
vec_pnl = df['pnl_daily'].values
pnl_mean = np.round(np.mean(vec_pnl) * 252, 4)
pnl_volatility = np.round(np.std(vec_pnl) * np.sqrt(252), 4)
sharpe = np.round(pnl_mean / pnl_volatility, 4)
df3 = df[df['trade'] == 1]
nb_trades = df3.shape[0]

# Plot Cumulative Pnl
plt_str = 'Cumulative PNL daily ' + str(backtest.ric_long) + ' / ' + str(backtest.ric_short) + '\n'\
        + 'pnl annual mean ' + str(pnl_mean) + '\n' \
        + 'pnl annual volatility ' + str(pnl_volatility) + '\n' \
        + 'pnl annual Sharpe ' + str(sharpe) + '\n' 
        
plt.figure()
plt.title(plt_str)
plt.xlabel('Time')
plt.ylabel('Cum PNL')
plt.plot(df['date'], df['cum_pnl_daily'])
plt.grid()
plt.show()