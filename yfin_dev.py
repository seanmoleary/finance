# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:36:18 2021

@author: seoleary
"""
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
 
ayx = yf.Ticker('AYX')
ayx.info
ayx.info.keys()
"""
history(self, period='1mo', interval='1d', start=None, end=None, prepost=False, actions=True, auto_adjust=True, back_adjust=False, proxy=None, rounding=False, tz=None, **kwargs)
Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
"""
close = ayx.history(period = '3mo', interval = '1d').Close
close.values.mean()
close.values.std()

ayx.options[0]
ayx.option_chain(ayx.options[0])

calls = ayx.option_chain(ayx.options[0]).calls

call_strikes = ayx.option_chain(ayx.options[0]).calls.strike

#Calculate the average price by averaging the bid/ask spread
price = (ayx.info['ask']+ayx.info['bid'])/2

a = {abs(strike-price):strike for strike in call_strikes}

closest_strike = a[min(a.keys())]

def get_closing_prices(ticker, period = '3mo', interval = '1d'):
    yf_data = yf.Ticker(ticker)
    history = yf_data.history(period = period, interval = interval)
    return history.Close
    
a = get_closing_prices('AYX')
plt.hist(a, int(round(len(a)/10,0)*5))

spy = get_closing_prices('SPY')
beta = np.cov(a,spy)[0,1]/np.var(a)

def beta(stock, spy):
    return np.cov(stock,spy)[0,1]/np.var(stock)

spy= get_closing_prices('spy', period = 'max',interval = '1d')
spy_df = pd.DataFrame(spy)
spy_df= spy_df.reset_index()
spy_df['dayofweek'] = spy_df['Date'].dt.day_name()
spy_df['month'] = spy_df['Date'].dt.month_name()
spy_df['year'] = spy_df.apply(lambda x: x.Date.year,axis=1)
spy_df['year']=spy_df.apply(lambda x: x.Date.year-min(spy_df.year),axis=1)
spy_df = pd.get_dummies(spy_df)
X = spy_df[['year', 'dayofweek_Friday', 'dayofweek_Monday',
       'dayofweek_Thursday', 'dayofweek_Tuesday', 'dayofweek_Wednesday',
       'month_April', 'month_August', 'month_December', 'month_February',
       'month_January', 'month_July', 'month_June', 'month_March', 'month_May',
       'month_November', 'month_October', 'month_September']]
Y = spy_df.Close
lr = LinearRegression()
lr.fit(X,Y)
[i for i in zip(X.columns, lr.coef_)]

X2 = sm.add_constant(X)
est = sm.OLS(Y,X2)
est2 = est.fit()
print(est2.summary())


{x:get_closing_prices(x,period = '5y', interval = '1d') for x in ['SPY','GOLD']}
beta = beta(a['GOLD'], a['SPY'])