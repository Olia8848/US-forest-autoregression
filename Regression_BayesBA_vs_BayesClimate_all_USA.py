# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:41:12 2020

@author: olga
"""


import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt
import random

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


# path = 'C:/Users/Olga Rumyantseva/Desktop/US_forest/' # home comp.
path = 'C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/'

# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'BayesSimsResults_all_USA.csv')
df.columns
# df = df.dropna()
data = df.values 
numpy.size(data, 0)


###########################################################################
#########   Regression BA vs 1 clim var : ######################################
###########################################################################


dataset = pandas.DataFrame({'Basal Area': data[:,0],
                            'AnnualMeanTemperature': data[:,1],
                            'MeanDiurnalRange': data[:,2],
                            'Isothermality': data[:,3],
                            'TemperatureSeasonality': data[:,4],
                            'MaxTemperatureofWarmestMonth': data[:,5],
                            'MinTemperatureofColdestMonth': data[:,6],
                            'TemperatureAnnualRange': data[:,7],
                            'MeanTemperatureofWettestQuarter': data[:,8],
                            'MeanTemperatureofDriestQuarter': data[:,9],
                            'MeanTemperatureofWarmestQuarter': data[:,10],
                            'MeanTemperatureofColdestQuarter': data[:,11],
                            'AnnualPrecipitation': data[:,12],
                            'PrecipitationofWettestMonth': data[:,13],
                            'PrecipitationofDriestMonth': data[:,14],
                            'PrecipitationSeasonality': data[:,15],
                            'PrecipitationofWettestQuarter': data[:,16],
                            'PrecipitationofDriestQuarter': data[:,17],
                            'PrecipitationofWarmestQuarte': data[:,18],
                            'PrecipitationofColdestQuarter': data[:,19]})
    

dataset.shape

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


print('clim. var ',numpy.argmax(R2results_for_clim_vars)+1,
      ' explains ',  round(numpy.max(R2results_for_clim_vars)*100, 1), '%')



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
    if j+1 == 15:
        continue
    X = dataset.iloc[:, [15,j+1]].values

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
    if (j+1 == 15 or j+1 == 9):
        continue
    X = dataset.iloc[:, [15, 9, j+1]].values

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
    if (j+1 == 2 or j+1 == 9 or j+1 == 15):
        continue
    X = dataset.iloc[:, [2, 9, 15, j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result4 = r2_score(y_test, y_pred)
    
    print(round(R2result4*100, 2))
#    print(j+1, round(R2result4*100, 2))
    R2results_for_clim_vars4 = numpy.append(R2results_for_clim_vars4, R2result4)


print(round(numpy.max(R2results_for_clim_vars4)*100, 3), '%')

###########################################################################
#########   Regression BA vs 5 clim vars : ######################################
###########################################################################

R2results_for_clim_vars5 = numpy.array([])


for j in range(19):
    if (j+1 == 2 or j+1 == 9 or j+1 == 15 or j+1 == 17):
        continue
    X = dataset.iloc[:, [2, 9, 15, 17, j+1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    R2result5 = r2_score(y_test, y_pred)
    
    print(round(R2result5*100, 2))
#    print(j+1, round(R2result4*100, 2))
    R2results_for_clim_vars5 = numpy.append(R2results_for_clim_vars5, R2result5)


print(round(numpy.max(R2results_for_clim_vars5)*100, 3), '%')

