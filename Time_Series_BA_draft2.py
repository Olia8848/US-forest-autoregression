# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:29:41 2020

@author: olga
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt

path = 'C:/Users/olga/Desktop/Time Series/'
data_1 = pd.read_csv(path + 'Ann_means_BA_232.csv')
data_1.columns

avg= data_1['Baavg']
avg=list(avg)
res = pd.Series(avg, index=pd.to_datetime(data_1['Year'],format='%Y'))

ts=res
ts_diff = ts - ts.shift()
ts_diff.dropna(inplace=True)
r = ARIMA(ts,(2,1,1))
r.plot_predict(1, 45)
r = r.fit(disp=-1)

pred = r.predict(1, 45)
dates = pd.date_range(1, 45)

predictions_ARIMA_diff = pd.Series(pred, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(ts.ix[0])
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(res)
plt.plot(predictions_ARIMA)

plt.show()

print predictions_ARIMA.head()
print ts.head()