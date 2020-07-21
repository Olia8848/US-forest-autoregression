# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:55:44 2019

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
import patsy


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
    
for i in range(NYears):
    logshadeTol = math.log(numpy.mean(shadeTol[year == Years[i]]))
    AnnualMeansShTol = numpy.append(AnnualMeansShTol, logshadeTol)


for i in range(NYears):
    c1 = math.log(numpy.mean(clim1[year == Years[i]]))
    AnnualMeansClim1 = numpy.append(AnnualMeansClim1, c1)

for i in range(NYears):
    c2 = math.log(numpy.mean(clim2[year == Years[i]]))
    AnnualMeansClim2 = numpy.append(AnnualMeansClim2, c2)

for i in range(NYears):
    c3 = math.log(numpy.mean(clim3[year == Years[i]]))
    AnnualMeansClim3 = numpy.append(AnnualMeansClim3, c3)

for i in range(NYears):
    c4 = math.log(numpy.mean(clim4[year == Years[i]]))
    AnnualMeansClim4 = numpy.append(AnnualMeansClim4, c4)

for i in range(NYears):
    c5 = math.log(numpy.mean(clim5[year == Years[i]]))
    AnnualMeansClim5 = numpy.append(AnnualMeansClim5, c5)    


a=numpy.array([Years, numpy.log(AnnualMeansBA), numpy.log(abs(AnnualMeansShTol)), 
   numpy.log(abs(AnnualMeansClim1)), numpy.log(abs(AnnualMeansClim2)), numpy.log(abs(AnnualMeansClim3)), 
   numpy.log(AnnualMeansClim4), numpy.log(AnnualMeansClim5)])



fig, axes = plt.subplots(nrows=4, ncols=2)
fig.tight_layout()
plt.subplot(3,2, 1)
plt.plot(Years, a[1, :], marker='', color='black',  label='kl')
plt.title('basal area', loc='left', fontsize=12, fontweight=0, color='black')
plt.subplot(3,2, 2)
plt.plot(Years, a[2, :], marker='', color='blue',  label='kl')
plt.title('shade tolerance index', loc='left', fontsize=12, fontweight=0, color='blue')
plt.subplot(3,2, 3)
plt.plot(Years, a[3, :], marker='', color='green',  label='kl')
plt.title('precip. seasonality', loc='left', fontsize=12, fontweight=0, color='green')
plt.subplot(3,2, 4)
plt.plot(Years, a[4, :], marker='', color='green',  label='kl')
plt.title('mean temp. of driest quarter', loc='left', fontsize=12, fontweight=0, color='green')
plt.subplot(3,2, 5)
plt.plot(Years, a[5, :], marker='', color='green',  label='kl')
plt.title('precip. of driest month', loc='left', fontsize=12, fontweight=0, color='green')
plt.subplot(3,2, 6)
plt.plot(Years, a[6, :], marker='', color='green',  label='kl')
plt.title('annual precip.', loc='left', fontsize=12, fontweight=0, color='green')


    
y=a[0, :]
data=pandas.DataFrame({'Year': y.astype(int), 
                       'AnnMeansBA': a[1, :],
                       'AnnMeansShTol': a[2, :], 
                       'AnnMeansClim1': a[3, :],
                       'AnnMeansClim2': a[4, :], 
                       'AnnMeansClim3': a[5, :],
                       'AnnMeansClim4': a[6, :], 
                       'AnnMeansClim5': a[7, :]})

data['Year'] = pandas.DatetimeIndex(data['Year'])
data.head()


#####################################################################

# null hypothesis is that the data are non-stationary
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['AnnMeansBA'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])  # 0.156157 > 0.05 - non-stationary

# mBAlog = numpy.log(data['AnnualMeansBA'])
diffBA = data['AnnualMeansBA'].diff(1)
diffBA = diffBA.dropna()
diffBA.plot()
plt.show()


########################################################

# !pip install C:\\Users\\olga\\Desktop\\pyflux-0.4.17-cp27-cp27m-win_amd64.whl

# import pyflux as pf


data1=data.diff(1)
data1=data1.dropna()
data1.head



model = pf.ARIMAX(data=data1, formula='AnnMeansBA~1+AnnMeansShTol+AnnMeansClim1+AnnMeansClim2+AnnMeansClim3+AnnMeansClim4+AnnMeansClim5',
                  ar=2, ma=2, integ=1, family=pf.Normal())
x = model.fit("MLE")
x.summary()


arimax.plot_diagnostics(figsize=(15, 12))

pred = x.get_prediction(start=pandas.to_datetime('1995-01-01'), dynamic=False)
pred_ci = pred.conf_int()













model_fit.plot_predict(dynamic=False)
plt.title('ARIMAX(2,1,2)')
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







residuals = pandas.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()


plot_pacf(residuals, lags=30)
plt.show()


qqplot(residuals, line='r', ax=pyplot.gca())
pyplot.show()

model_fit.plot_predict(dynamic=False)
plt.show()
