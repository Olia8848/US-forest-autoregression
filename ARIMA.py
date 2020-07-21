# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:32:46 2020

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
from matplotlib import pyplot as plt

path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
data = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
data.columns
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

# Annual means of shadows:
AnnualMeansBA = numpy.array([]) 
AnnualMeansShTol = numpy.array([])
AnnualMeansClim1 = numpy.array([])
AnnualMeansClim2 = numpy.array([])
AnnualMeansClim3 = numpy.array([])
AnnualMeansClim4 = numpy.array([])
AnnualMeansClim5 = numpy.array([])



for i in range(NYears):
    logBA = math.log(numpy.mean(barea[year == Years[i]]))
    AnnualMeansBA = numpy.append(AnnualMeansBA, logBA)


a=numpy.array([Years, numpy.log(numpy.log(AnnualMeansBA))])

y=a[0, :]
data=pandas.DataFrame({'Year': y.astype(int), 
                       'AnnualMeansBA': a[1, :]})



data['Year'] = pandas.DatetimeIndex(data['Year'])
data.head()


fig1 = plt.figure()
pyplot.plot(Years, AnnualMeansBA, 'bo', Years, AnnualMeansBA, 'k')
#plt.fill_between(Years, AnnualMeansBA, color='green')
pyplot.show()



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



import statsmodels
from statsmodels.tsa.arima_model import ARIMA

# Diff = diff.to_frame().dropna().as_matrix()
#                          p, d, q
BA = data['AnnualMeansBA'].to_frame().dropna().as_matrix()
model = ARIMA(BA, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

model_fit.plot_predict(dynamic=False)
plt.title('ARIMA(1,1,1)')
plt.legend(['model prediction', 'Basal Area'])
#plt.show()

#residuals = pandas.DataFrame(model_fit.resid)
#fig, ax = plt.subplots(1,2)
#residuals.plot(title="residuals", ax=ax[0])
#residuals.plot(kind='kde', title='residuals density', ax=ax[1])
#plt.show()

res = model_fit.resid # residuals
fig = sm.qqplot(res, fit=True, line='45')
plt.show()

