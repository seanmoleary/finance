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