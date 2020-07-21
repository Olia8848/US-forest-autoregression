# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:13:36 2020

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


def inverseGamma(alpha, beta):
    return (1/numpy.random.gamma(alpha, 1/beta))

def getClimaticFactorsforRegress(data, NSIMS):
    
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

    return (bio1, bio2, bio3, bio4,
            bio5, bio6, bio7, bio8, 
            bio9, bio10, bio11, bio12, 
            bio13, bio14, bio15, bio16, 
            bio17, bio18, bio19)



def BayesVect(NSIMS, vec):
    path = 'C:/Users/olga/Desktop/US_forest/'
    df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
    df = df.drop('Unnamed: 0', 1)
    df = df.dropna()
    data = df.values 
    NOBSERV = numpy.size(data, 0)
    patch = data[:,21]   # Plot ID 
    year = numpy.array([int(data[k,22]) for k in range(NOBSERV)]) # Year 
    
    Years = numpy.unique(year)
    NYears = numpy.size(Years)
    Patches  = numpy.unique(patch)
    NPatches = numpy.size(Patches)
    
    LogValuePatchYear = [[0] * NPatches for i in range(NYears)]  
    for k in range(NOBSERV):
        indY = numpy.where(Years == year[k])
        i = indY[0][0]     # return i such that year[k] == Years[i]
        indP = numpy.where(Patches == patch[k])
        j = indP[0][0]  # return j such that patch[k] == Patches[j]
        LogValuePatchYear[i][j] = vec[k]  
#        LogValuePatchYear[i][j] = math.log(vec[k])
    
    meansB =  [[0] * NSIMS for i in range(NYears)] 
    varsB = [[0] * NSIMS for i in range(NYears)]
    
    for t in range(NYears): 
        LogValuesYear = numpy.array([])  # biomasses vector x_{1}(t), ... , x_{#p}(t)
        for p in range(NPatches):    
            LogValuesYear = numpy.append(LogValuesYear, LogValuePatchYear[t][p])
        LogValuesYear = LogValuesYear[numpy.nonzero(LogValuesYear)] #  biomass data doesn't  contain 0 records, so we can remove zeroes:
    
        empMean = numpy.mean(LogValuesYear) # mean of biomasses in year t
        empVar = numpy.var(LogValuesYear) # variance of biomasses in year t
    
        n = len(LogValuesYear) # number of patches observed in year t
        if n == 1:
            for j in range(NSIMS):
               varsB[t][j] = empVar 
               meansB[t][j] = empMean
    
        if n != 1:
            for j in range(NSIMS):
                varsB[t][j] = inverseGamma((n-1)/2, (n*empVar)/2) 
                meansB[t][j] = random.normal(empMean, varsB[t][j]/n)    
    
    vecBayes = numpy.array([]) 
    for n in range(NSIMS):
         for t in range(NYears):
              b = meansB[t][n]
              vecBayes = numpy.append(vecBayes, b)
    
    return (vecBayes)


###########################################################################
path = 'C:/Users/olga/Desktop/US_forest/'
# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df = df.drop('Unnamed: 0', 1)
# df.columns
df = df.dropna()
data = df.values 

biomass = data[:,23] # Biomass
barea = data[:,24]   # Basal Area 

NSIMS = 1000

# BAsBayes = BayesVect(NSIMS, barea)
# pandas.DataFrame(BAsBayes).to_csv("C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/BayesSimsResults.csv")


# now change remove ln from function
bio = getClimaticFactorsforRegress(data, NSIMS)

bio1Bayes = BayesVect(NSIMS, bio[0])
bio1Bayes = pandas.DataFrame(bio1Bayes)
BAsBay = pandas.read_csv('C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/BayesSimsResults.csv')
BAsBay = pandas.DataFrame(BAsBay)
a = pandas.concat([BAsBay, bio1Bayes], axis=1)
pandas.DataFrame(a).to_csv("C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/BayesSimsResults.csv")


bio2Bayes = BayesVect(NSIMS, bio[1])
bio2Bayes = pandas.DataFrame(bio2Bayes)
BAsBay = pandas.read_csv('C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/BayesSimsResults.csv')
BAsBay = pandas.DataFrame(BAsBay)
a = pandas.concat([BAsBay, bio2Bayes], axis=1)
pandas.DataFrame(a).to_csv("C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/BayesSimsResults.csv")


for j in range(10, 19):
    print(j)     
    bioB = BayesVect(NSIMS, bio[j])
    bioB = pandas.DataFrame(bioB)
    BAsBay = pandas.read_csv('C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/BayesSimsResults.csv')
    BAsBay = pandas.DataFrame(BAsBay)
    a = pandas.concat([BAsBay, bioB], axis=1)
    pandas.DataFrame(a).to_csv("C:/Users/olga/Desktop/US_forest/Bayes_vectors_for_regression/BayesSimsResults.csv")



###########################################################################
#########   Regression BA vs 1 clim var : ######################################
###########################################################################

dataset = pandas.DataFrame({'Basal Area': BAsBayes,
                            'AnnualMeanTemperature': bio1Bayes,
                            'MeanDiurnalRange': bio[1],
                            'Isothermality': bio[2],
                            'TemperatureSeasonality': bio[3],
                            'MaxTemperatureofWarmestMonth': bio[4],
                            'MinTemperatureofColdestMonth': bio[5],
                            'TemperatureAnnualRange': bio[6],
                            'MeanTemperatureofWettestQuarter': bio[7],
                            'MeanTemperatureofDriestQuarter': bio[8],
                            'MeanTemperatureofWarmestQuarter': bio[9],
                            'MeanTemperatureofColdestQuarter': bio[10],
                            'AnnualPrecipitation': bio[11],
                            'PrecipitationofWettestMonth': bio[12],
                            'PrecipitationofDriestMonth': bio[13],
                            'PrecipitationSeasonality': bio[14],
                            'PrecipitationofWettestQuarter': bio[15],
                            'PrecipitationofDriestQuarter': bio[16],
                            'PrecipitationofWarmestQuarte': bio[17],
                            'PrecipitationofColdestQuarter': bio[18]})
    

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

