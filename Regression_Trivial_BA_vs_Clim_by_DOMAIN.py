# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:17:19 2019

@author: olga
"""


import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt
import random
import docx

from numpy import random
from numpy import mean
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


path = 'C:/Users/olga/Desktop/US_forest/'

# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df.columns

df = df.drop('Unnamed: 0', 1)
df.columns



data0 = df.values  
NOBSERV0 = numpy.size(data0, 0) 


ecoreg = numpy.array([str(data0[k,25]) for k in range(NOBSERV0)])  # Eco region
Ecoregions  = numpy.unique(ecoreg)


Humid_200 = ['211', '212', '221', '222', '223', '231', '232', '234', '242',
       '251', '255', '261', '262', '263',  'M211', 'M221', 'M223', 'M231',
       'M242', 'M261', 'M262']

Dry_300 = ['313', '315', '321', '322',
       '331', '332', '341', '342', 'M313', 'M331', 'M332', 'M333', 'M334',
       'M341']

Humid_Tropical_400 = ['411']


Domain = Humid_Tropical_400

df2 = df[numpy.isin(ecoreg, Domain)]

data = df2.values


bio1 = data[:,2]  # AnnualMeanTemperature
bio2 = data[:,3]  # MeanDiurnalRange
bio3 = data[:,4]  # Isothermality
bio4 = data[:,5]  # TemperatureSeasonality
bio5 = data[:,6]  # MaxTemperatureofWarmestMonth
bio6 = data[:,7]  # MinTemperatureofColdestMonth
bio7 = data[:,8]  # TemperatureAnnualRange
bio8 = data[:,9]  # MeanTemperatureofWettestQuarter
bio9 = data[:,10]  # MeanTemperatureofDriestQuarter
bio10 = data[:,11]  # MeanTemperatureofWarmestQuarter
bio11 = data[:,12]  # MeanTemperatureofColdestQuarter
bio12 = data[:,13]  # AnnualPrecipitation
bio13 = data[:,14]  # PrecipitationofWettestMonth
bio14 = data[:,15]  # PrecipitationofDriestMonth
bio15 = data[:,16]  # PrecipitationSeasonality
bio16 = data[:,17]  # PrecipitationofWettestQuarter
bio17 = data[:,18]  # PrecipitationofDriestQuarter
bio18 = data[:,19]  # PrecipitationofWarmestQuarte
bio19 = data[:,20]  # PrecipitationofColdestQuarter

biomass = data[:,23] # Biomass
barea = data[:,24]   # Basal Area


value = barea


###########################################################################
#########   Regression BA vs 1 clim var : ######################################
###########################################################################


dataset = pandas.DataFrame({'Basal Area':value,
                            'AnnualMeanTemperature': bio1,
                            'MeanDiurnalRange': bio2,
                            'Isothermality': bio3,
                            'TemperatureSeasonality': bio4,
                            'MaxTemperatureofWarmestMonth': bio5,
                            'MinTemperatureofColdestMonth': bio6,
                            'TemperatureAnnualRange': bio7,
                            'MeanTemperatureofWettestQuarter': bio8,
                            'MeanTemperatureofDriestQuarter': bio9,
                            'MeanTemperatureofWarmestQuarter': bio10,
                            'MeanTemperatureofColdestQuarter': bio11,
                            'AnnualPrecipitation': bio12,
                            'PrecipitationofWettestMonth': bio13,
                            'PrecipitationofDriestMonth': bio14,
                            'PrecipitationSeasonality': bio15,
                            'PrecipitationofWettestQuarter': bio16,
                            'PrecipitationofDriestQuarter': bio17,
                            'PrecipitationofWarmestQuarte': bio18,
                            'PrecipitationofColdestQuarter': bio19})
    

dataset.shape

dataset = dataset.fillna(method='ffill')
dataset[dataset == numpy.inf] = numpy.nan
dataset.fillna(dataset.mean(), inplace=True)


y = dataset[dataset.columns[0]].values  # dataset['Basal Area']



R2results_for_clim_vars = numpy.array([])

for j in range(19):
    X = dataset[dataset.columns[j+1]].values
    # split 80% of the data to training set while 
    # 20% of the data to test set:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train= X_train.reshape(-1, 1)
    y_train= y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result = r2_score(y_test, y_pred)
    print(round(R2result*100, 2))
#    print(j+1, round(R2result*100, 2))
    R2results_for_clim_vars = numpy.append(R2results_for_clim_vars, R2result)


print(round(numpy.max(R2results_for_clim_vars)*100, 1), '%')



# Plot outputs
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()



###########################################################################
#########   Regression BA vs 2 clim vars : ######################################
###########################################################################

y = dataset[dataset.columns[0]].values  # dataset['Basal Area']

R2results_for_clim_vars2 = numpy.array([])


for j in range(19):
    if j+1 == 3:
        continue
    X = dataset.iloc[:, [3,j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result2 = r2_score(y_test, y_pred)
    
    print(round(R2result2*100, 3))
#    print(j+1, round(R2result2*100, 2))
    R2results_for_clim_vars2 = numpy.append(R2results_for_clim_vars2, R2result2)


print(round(numpy.max(R2results_for_clim_vars2)*100, 2), '%')

###########################################################################
#########   Regression BA vs 3 clim vars : ######################################
###########################################################################

R2results_for_clim_vars3 = numpy.array([])


for j in range(19):
    if (j+1 == 3 or j+1 == 12):
        continue
    X = dataset.iloc[:, [3, 12, j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result3 = r2_score(y_test, y_pred)
    
    print(round(R2result3*100, 3))
#    print(j+1, round(R2result3*100, 2))
    R2results_for_clim_vars3 = numpy.append(R2results_for_clim_vars3, R2result3)


print(round(numpy.max(R2results_for_clim_vars3)*100, 2), '%')


###########################################################################
#########   Regression BA vs 4 clim vars : ######################################
###########################################################################

R2results_for_clim_vars4 = numpy.array([])


for j in range(19):
    if (j+1 == 3 or j+1 == 12 or j+1 == 5):
        continue
    X = dataset.iloc[:, [3, 12, 5, j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result4 = r2_score(y_test, y_pred)
    
    print(round(R2result4*100, 2))
#    print(j+1, round(R2result4*100, 2))
    R2results_for_clim_vars4 = numpy.append(R2results_for_clim_vars4, R2result4)


print(round(numpy.max(R2results_for_clim_vars4)*100, 3), '%')

