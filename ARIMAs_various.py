# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:13:36 2020

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
from matplotlib import pyplot as plt

path = 'C:/Users/olga/Desktop/US_forest/'
data = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
data.columns
data = data.iloc[:, [22,23,24,25,26,27,1,2,17,11,16,14,21]]

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

#for j in range(NEcoregions):
#    print(j, Ecoregions[j])


eid = 0

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
clim1 = data1[:,8]   # PrecipitationSeasonality
clim2 = data1[:,9]   # MeanTemperatureofDriestQuarter
clim3 = data1[:,10]   # PrecipitationofDriestMonth
clim4 = data1[:,11]   # AnnualPrecipitation
clim5 = data1[:,12]   # AnnualPrecipitation

Years = numpy.unique(year)
NYears = numpy.size(Years)

AnnualMeansBA = numpy.array([]) 
for i in range(NYears):
    logBA = math.log(numpy.mean(barea[year == Years[i]]))
    AnnualMeansBA = numpy.append(AnnualMeansBA, logBA)

a=numpy.array([Years, numpy.log(numpy.log(AnnualMeansBA))])
y=a[0, :]
data=pandas.DataFrame({'Year': y.astype(int), 
                       'AnnualMeansBA': a[1, :]})
data['Year'] = pandas.DatetimeIndex(data['Year'])
data.head()




# null hypothesis is that the data are non-stationary
from statsmodels.tsa.stattools import adfuller
result = adfuller(AnnualMeansBA)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])  # 0.156157 > 0.05 - non-stationary

numpy.size(AnnualMeansBA) # 33+
# mBAlog = numpy.log(data['AnnualMeansBA'])
diff = data['AnnualMeansBA'].diff(1)
diff = diff.dropna()
diff.plot()
plt.show()
# so choose d=1
result = adfuller(diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1]) # p-value: 0.000000 stationary



fig1 = plt.figure()
pyplot.plot(Years, AnnualMeansBA, 'bo', Years, AnnualMeansBA, 'k')
#plt.fill_between(Years, AnnualMeansBA, color='green')
pyplot.show()



import statsmodels 
from statsmodels.tsa.arima_model import ARIMA

#                          p, d, q
BA = data['AnnualMeansBA'].to_frame().dropna().as_matrix()
model1 = ARIMA(BA, order=(1,0,0))
model2 = ARIMA(BA, order=(0,1,0))
model3 = ARIMA(BA, order=(0,0,1))
model4 = ARIMA(BA, order=(1,1,0))
model5 = ARIMA(BA, order=(1,0,1))
model6 = ARIMA(BA, order=(1,1,0))
model7 = ARIMA(BA, order=(1,1,1))

model1_fit = model1.fit()
# print(model_fit.summary())
model1_fit.plot_predict(dynamic=False)
plt.title('ARIMA(1,0,0)')
plt.legend(['model prediction', 'basal area'])
res = model1_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')

model2_fit = model2.fit()
# print(model_fit.summary())
model2_fit.plot_predict(dynamic=False)
plt.title('ARIMA(0,1,0)')
plt.legend(['model prediction', 'basal area'])
res = model2_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')


model3_fit = model3.fit()
# print(model_fit.summary())
model3_fit.plot_predict(dynamic=False)
plt.title('ARIMA(0,0,1)')
plt.legend(['model prediction', 'basal area'])
res = model3_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')


model4_fit = model4.fit()
# print(model_fit.summary())
model4_fit.plot_predict(dynamic=False)
plt.title('ARIMA(1,1,0)')
plt.legend(['model prediction', 'basal area'])
res = model4_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')


model5_fit = model5.fit()
# print(model_fit.summary())
model5_fit.plot_predict(dynamic=False)
plt.title('ARIMA(1,0,1)')
plt.legend(['model prediction', 'basal area'])
res = model5_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')


model6_fit = model6.fit()
# print(model_fit.summary())
model6_fit.plot_predict(dynamic=False)
plt.title('ARIMA(0,1,1)')
plt.legend(['model prediction', 'basal area'])
res = model6_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')


model7_fit = model7.fit()
# print(model_fit.summary())
model7_fit.plot_predict(dynamic=False)
plt.title('ARIMA(1,1,1)')
plt.legend(['model prediction', 'basal area'])
res = model7_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')
plt.show()

