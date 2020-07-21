# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:46:16 2019

@author: olga
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:40:26 2019

@author: Olga Rumyantseva
"""


import pandas   # data analysis
import numpy    # n-dim arrays
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import scipy    # lin regress
import matplotlib.pyplot as plt
#from pandas.tools.plotting import table
import random

##################################################################
#################### Preparation Block: ##########################
##################################################################


path = 'C:/Users/olga/Desktop/Quebec/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'

# df = pandas.read_csv(path + 'biodata.csv')
df = pandas.read_csv(path + 'biodata.csv') # header added as the 
# df.columns

data = df.values  # work with values only
NOBSERV = numpy.size(data, 0) # number of rows in our dataframe (number of observations)
print(NOBSERV) 


patch = data[:,0]   # plot ID (1st column)
year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) #years converted in integer from float
biomass = data[:,2]
shadow = data[:,3]
value = biomass # this is what we consider now: biomass or shadow


##################################################################
#####  Annual means of biomass and shadow: ########################
##################################################################


Years = numpy.unique(year)
NYears = numpy.size(Years)

# Annual means of biomass:
AnnualMeanBiomass = numpy.array([]) 

for i in range(NYears):
    logbio = math.log(numpy.mean(value[year == Years[i]]))
    AnnualMeanBiomass = numpy.append(AnnualMeanBiomass, logbio)
    

numpy.size(AnnualMeanBiomass)
# AnnualMeanBiomass = numpy.repeat(AnnualMeanBiomass, 2)
# Years = range(1970,2046,1)  
 
fig1 = plt.figure()
pyplot.plot(Years, AnnualMeanBiomass, 'bo', Years, AnnualMeanBiomass, 'k')
pyplot.show()


NYears


a=numpy.array([Years, numpy.log(AnnualMeanBiomass)])

y=a[0]
data=pandas.DataFrame({'Year': y.astype(int), 
                       'AnnualMeanBiomass': a[1]})


data['Year'] = pandas.DatetimeIndex(data['Year'])
data.head()


# null hypothesis is that the data are non-stationary
from statsmodels.tsa.stattools import adfuller
result = adfuller(AnnualMeanBiomass)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])  # 0.156157 > 0.05 - non-stationary


# mBAlog = numpy.log(data['AnnualMeansBA'])
diff = data['AnnualMeanBiomass'].diff(1)
diff = diff.dropna()
diff.plot()
plt.show()


# so choose d=1
result = adfuller(diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1]) # p-value: 0.000000 stationary


import statsmodels.api
plot_acf(data['AnnualMeanBiomass'], lags=30)
plot_pacf(data['AnnualMeanBiomass'], lags=30)
plt.show()

plot_acf(diff, lags=30)
plot_pacf(diff, lags=30)
plt.show()

for i in range(NYears):
    print(i, AnnualMeanBiomass[i], Years[i])
    
AnnualMeanBiomass[20] = 2.4   
