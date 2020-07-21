# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:05:24 2019

@author: olga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:14:28 2019

@author: olga
"""

import pandas   # data analysis
import numpy    # n-dim arrays
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import scipy    # lin regress
import random
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
data = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
data.columns
# precip. seasonality,
# mean temp of driest quart
# precip of driest month
# annual precip
# precip of coldest quarter

data = data.iloc[:, [22,23,24,25,26,27,1,2,17,11,16,14,21]]
data.columns


data0 = data.values  
NOBSERV0 = numpy.size(data, 0) 
print(NOBSERV0) #  number of observations totally 409 868

patch = data0[:,0]   # Plot ID 
year = numpy.array([int(data0[k,1]) for k in range(NOBSERV0)]) # Year 
biomass = data0[:,2] # Biomass
barea = data0[:,3]   # Basal Area
shadeTol = data0[:,4]   # Basal Area
ecoreg = numpy.array([str(data0[k,5]) for k in range(NOBSERV0)])  # Eco region
clim1 = data0[:,8]   # PrecipitationSeasonality
clim2 = data0[:,9]   # MeanTemperatureofDriestQuarter
clim3 = data0[:,10]   # PrecipitationofDriestMonth
clim4 = data0[:,11]   # AnnualPrecipitation
clim5 = data0[:,12]   # AnnualPrecipitation


Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions) # 36 ecoregions


Years = numpy.unique(year)
NYears = numpy.size(Years)

# Annual means of shadows:
AnnualMeansBA = numpy.array([]) 
AnnualMeansShTol = numpy.array([])
AnnualMeansClim1 = numpy.array([])
AnnualMeansClim2 = numpy.array([])
AnnualMeansClim3 = numpy.array([])
AnnualMeansClim4 = numpy.array([])
AnnualMeansClim5 = numpy.array([])



for i in range(NYears):
    logBA = numpy.log(numpy.log(numpy.log(numpy.mean(barea[year == Years[i]]))))
    AnnualMeansBA = numpy.append(AnnualMeansBA, logBA)
    
for i in range(NYears):
    logshadeTol = numpy.mean(shadeTol[year == Years[i]])
    AnnualMeansShTol = numpy.append(AnnualMeansShTol, logshadeTol)


for i in range(NYears):
    c1 = numpy.mean(clim1[year == Years[i]])
    AnnualMeansClim1 = numpy.append(AnnualMeansClim1, c1)

for i in range(NYears):
    c2 = numpy.mean(clim2[year == Years[i]])
    AnnualMeansClim2 = numpy.append(AnnualMeansClim2, c2)

for i in range(NYears):
    c3 = numpy.mean(clim3[year == Years[i]])
    AnnualMeansClim3 = numpy.append(AnnualMeansClim3, c3)    
    
for i in range(NYears):
    c4 = numpy.mean(clim4[year == Years[i]])
    AnnualMeansClim4 = numpy.append(AnnualMeansClim4, c4)

for i in range(NYears):
    c5 = numpy.mean(clim5[year == Years[i]])
    AnnualMeansClim5 = numpy.log(numpy.append(AnnualMeansClim5, c5))    



a=numpy.array([Years, 
               AnnualMeansBA, 
               AnnualMeansShTol, 
               AnnualMeansClim1, 
               AnnualMeansClim2, 
               AnnualMeansClim3, 
               AnnualMeansClim4, 
               AnnualMeansClim5])

y=a[0]
data=pandas.DataFrame({'Year': y.astype(int), 
                       'AnnualMeansBA': a[1],
                       'AnnualMeansShTol': a[2], 
                       'AnnualMeansClim1': a[3],
                       'AnnualMeansClim2': a[4], 
                       'AnnualMeansClim3': a[5],
                       'AnnualMeansClim4': a[6], 
                       'AnnualMeansClim5': a[7]})



data['Year'] = pandas.DatetimeIndex(data['Year'])
data.head()


fig1 = plt.figure()
pyplot.plot(Years, AnnualMeansBA, 'bo', Years, AnnualMeansBA, 'k')
pyplot.show()

fig1 = plt.figure()
pyplot.plot(Years, AnnualMeansClim3, 'bo', Years, AnnualMeansClim3, 'k')
pyplot.show()


from statsmodels.tsa.stattools import adfuller
result = adfuller(AnnualMeansClim3)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# 3 PrecipitationSeasonality
# 4 MeanTemperatureofDriestQuarter
# 5 PrecipitationofDriestMonth
# 6 AnnualPrecipitation
# 7 AnnualPrecipitation

data1=pandas.DataFrame({'AnnualMeansBA': a[5], 
                       'AnnualMeansClim3': a[1]})
from statsmodels.tsa.stattools import grangercausalitytests
granger_test_result = sm.tsa.stattools.grangercausalitytests(data1, maxlag=12, verbose=True)


# at lag 2  AnnualMeansBA = f(AnnualMeansClim3) cannot be rejected.

numpy.size(AnnualMeansBA)

v1 = numpy.mean(AnnualMeansBA)-numpy.min(AnnualMeansBA)
v2 = numpy.mean(AnnualMeansClim3)-numpy.min(AnnualMeansClim3)

b=AnnualMeansBA-numpy.mean(AnnualMeansBA)
cl =AnnualMeansClim3-numpy.mean(AnnualMeansClim3)



fig1 = plt.figure()
pyplot.plot(Years, b*25+1, 'b', Years, b*25+1, 'green',linewidth=2.5)
pyplot.plot(Years, cl, 'b', Years, cl, 'blue',linewidth=2)
plt.legend({'basal area','precip of driest month'})
plt.grid()
pyplot.show()


plt.scatter( cl, b*25+1, marker='o')

data1=pandas.DataFrame({'BA': b*25+1, 
                       'Clim3': cl})

from statsmodels.tsa.stattools import grangercausalitytests
granger_test_result = sm.tsa.stattools.grangercausalitytests(data1, maxlag=12, verbose=True)














data1=pandas.DataFrame({'AnnualMeansBA': b*25+1, 
                       'AnnualMeansClim3': cl})
train = data1[:int(0.8*(len(data1)))]
valid = data1[int(0.8*(len(data1))):]
#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = sm.tsa.VARMAX(train, order=(0,1))
# (p,q) order AR and MA 
model_result = model.fit(maxiter=1000, disp=False)
print('AIC: ', model_result.aic)
model_result.plot_diagnostics()
model_result.forecast(steps=4)
    
 




   
for i in range(3):
    i += 1
    model = sm.tsa.VARMAX(train, order=(i,0))
    model_result = model.fit(maxiter=1000, disp=False)
    print('q=0, p = ', i)
    print('AIC: ', model_result.aic)
    model = sm.tsa.VARMAX(train, order=(i,1))
    model_result = model.fit(maxiter=1000, disp=False)
    print('q=1, p = ', i)
    print('AIC: ', model_result.aic)
    model = sm.tsa.VARMAX(train, order=(i,2))
    model_result = model.fit(maxiter=1000, disp=False)
    print('q=2, p = ', i)
    print('AIC: ', model_result.aic)


model = sm.tsa.VARMAX(train, order=(1,1))
model_result = model.fit(maxiter=1000, disp=False)
print('q=0, p = ', i)
print('AIC: ', model_result.aic)

model_result.plot_diagnostics()
model_result.forecast(steps=4)















fig, axes = plt.subplots(nrows=4, ncols=2)
fig.tight_layout()
plt.subplot(3,2, 1)
plt.plot(Years, AnnualMeansBA, marker='', color='black',  label='kl')
plt.title('basal area', loc='left', fontsize=12, fontweight=0, color='black')
plt.subplot(3,2, 2)
plt.plot(Years, AnnualMeansShTol, marker='', color='blue',  label='kl')
plt.title('shade tolerance index', loc='left', fontsize=12, fontweight=0, color='blue')
plt.subplot(3,2, 3)
plt.plot(Years, AnnualMeansClim1, marker='', color='green',  label='kl')
plt.title('precip. seasonality', loc='left', fontsize=12, fontweight=0, color='green')
plt.subplot(3,2, 4)
plt.plot(Years, AnnualMeansClim2, marker='', color='green',  label='kl')
plt.title('mean temp. of driest quarter', loc='left', fontsize=12, fontweight=0, color='green')
plt.subplot(3,2, 5)
plt.plot(Years, AnnualMeansClim3, marker='', color='green',  label='kl')
plt.title('precip. of driest month', loc='left', fontsize=12, fontweight=0, color='green')
plt.subplot(3,2, 6)
plt.plot(Years, AnnualMeansClim4, marker='', color='green',  label='kl')
plt.title('annual precip.', loc='left', fontsize=12, fontweight=0, color='green')



# null hypothesis is that the data are non-stationary
from statsmodels.tsa.stattools import adfuller
result = adfuller(AnnualMeansBA)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])  # 0.156157 > 0.05 - non-stationary


# mBAlog = numpy.log(data['AnnualMeansBA'])
diff = data['AnnualMeansBA'].diff(1)
diff = diff.dropna()
diff.plot()
plt.show()


# so choose d=1
result = adfuller(diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1]) # p-value: 0.000000 stationary


########################################################
# pip install --upgrade --no-deps statsmodels
import statsmodels.api

AA=statsmodels.tsa.stattools.pacf(diff, nlags=30)
numpy.argsort(numpy.abs(AA))

#####################################################
 
plot_acf(diff, lags=30)
plot_pacf(diff, lags=30)
plt.show()

plot_acf(data['AnnualMeansBA'], lags=30)
plot_pacf(data['AnnualMeansBA'], lags=30)
plt.show()



import statsmodels
from statsmodels.tsa.arima_model import ARIMA

Diff = diff.to_frame().dropna().as_matrix()

model = ARIMA(Diff, order=(1,1,1))
#                         (p-AR, d-diff, q-MA)
#                         1, 23    1     1,3,6
model_fit = model.fit()
print(model_fit.summary())

model_fit.plot_predict(dynamic=False)
plt.title('ARIMA(1,1,1)')
plt.legend(['model prediction', 'Basal Area'])
plt.show()

residuals = pandas.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="residuals", ax=ax[0])
residuals.plot(kind='kde', title='residuals density', ax=ax[1])
plt.show()

fig, ax = plt.subplots(1,2)
plot_acf(residuals, lags=30, ax=ax[0], title='residuals ACF')
plot_pacf(residuals, lags=30, ax=ax[1], title='residuals PACF')
plt.show()

qqplot(residuals, line='r', ax=pyplot.gca())
pyplot.title('residuals QQ plot')
pyplot.show()










################################
numpy.size(diff)

train = diff[:20]
test = diff[20:]

model = ARIMA(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1)  
# Forecast
fc, se, conf = fitted.forecast(12, alpha=0.05)  # 95% conf
# Make as pandas series
fc_series = pandas.Series(fc, index=test.index)
lower_series = pandas.Series(conf[:, 0], index=test.index)
upper_series = pandas.Series(conf[:, 1], index=test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


model = ARIMA(train, order=(23, 1, 6))  
fitted = model.fit(disp=-1)  
# Forecast
fc, se, conf = fitted.forecast(12, alpha=0.05)  # 95% conf
# Make as pandas series
fc_series = pandas.Series(fc, index=test.index)
lower_series = pandas.Series(conf[:, 0], index=test.index)
upper_series = pandas.Series(conf[:, 1], index=test.index)
# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm

model = pm.auto_arima(diff, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=6, # maximum p and q
                      m=1,              # frequency of series
                      d=1,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=1, 
                      D=1, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())



