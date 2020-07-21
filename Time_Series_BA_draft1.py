# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:49:42 2020

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

###########################################################################
#########   Restricted by ecoregion: ######################################
###########################################################################

eid = 6

data = data[ecoreg == Ecoregions[eid]]
data.columns

data1 = data.values  

NOBSERV = numpy.size(data1, 0) 
print(NOBSERV)  # observations in your ecoregion

patch = data1[:,0]   # Plot ID 
year = numpy.array([int(data1[k,1]) for k in range(NOBSERV)]) # Year 
biomass = data1[:,2] # Biomass
barea = data1[:,3]   # Basal Area
shadeTol = data1[:,4]   # Basal Area
ecoreg = numpy.array([str(data1[k,5]) for k in range(NOBSERV)])  # Eco region


Years = numpy.unique(year)
NYears = numpy.size(Years)

# Annual means of shadows:
AnnualMeansBA = numpy.array([]) 


for i in range(NYears):
    logBA = math.log(numpy.mean(barea[year == Years[i]]))
    AnnualMeansBA = numpy.append(AnnualMeansBA, logBA)

for i in range(NYears):
    print(AnnualMeansBA[i])

a=numpy.array([Years, numpy.log(AnnualMeansBA)])

y=a[0, :]
data=pandas.DataFrame({'Year': y.astype(int), 
                       'AnnualMeansBA': a[1, :]})



data['Year'] = pandas.DatetimeIndex(data['Year'])
data.head()


fig1 = plt.figure()
pyplot.plot(Years, AnnualMeansBA, 'bo', Years, AnnualMeansBA, 'k')
pyplot.show()


diff = data['AnnualMeansBA'].diff(1)
diff = diff.dropna()
diff.plot()
plt.show()

import statsmodels
from statsmodels.tsa.arima_model import ARIMA

Diff = diff.to_frame().dropna().as_matrix()

numpy.size(Diff)

r = ARIMA(Diff, order=(2, 1, 1))
r = r.fit(disp=-1)
r.plot_predict(1, 45)
pred = r.predict(1, 45)


#dates = pd.date_range('1961-01','1970-01',freq='M')

predictions_ARIMA_diff = pandas.Series(pred, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pandas.Series(Diff.ix[0])

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA = numpy.exp(predictions_ARIMA_log)
plt.plot(res)
plt.plot(predictions_ARIMA)

plt.show()

print predictions_ARIMA.head()
print ts.head()












model = ARIMA(Diff, order=(2,1,1))
#                         (p-AR, d-diff, q-MA)
#                         1, 23    1     1,3,6
model_fit = model.fit()
print(model_fit.summary())

model_fit.plot_predict('1990', '2050', dynamic=False)
plt.title('ARIMA(2,1,1)')
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



