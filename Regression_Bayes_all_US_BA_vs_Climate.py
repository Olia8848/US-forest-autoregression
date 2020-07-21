# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:42:44 2019

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



def inverseGamma(alpha, beta):
    return (1/numpy.random.gamma(alpha, 1/beta))

def getClimaticFactorsforRegress(data, NSIMS):

    NOBSERV = numpy.size(data, 0)
   
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
    
    year = numpy.array([int(data[k,22]) for k in range(NOBSERV)]) # Year 
    Years = numpy.unique(year)
    NYears = numpy.size(Years)

# organize them by year:
    bio1eco = numpy.array([]) # AnnualMeanTemperature
    for t in range(NYears):
        b = numpy.mean(bio1[year == Years[t]])
        bio1eco = numpy.append(bio1eco, b)
        
    bio2eco = numpy.array([])  # MeanDiurnalRange
    for t in range(NYears):
        b = numpy.mean(bio2[year == Years[t]])
        bio2eco = numpy.append(bio2eco, b)
        
    bio3eco = numpy.array([])  # Isothermality
    for t in range(NYears):
        b = numpy.mean(bio3[year == Years[t]])
        bio3eco = numpy.append(bio3eco, b)
        
    bio4eco = numpy.array([])  # TemperatureSeasonality
    for t in range(NYears):
        b = numpy.mean(bio4[year == Years[t]])
        bio4eco = numpy.append(bio4eco, b)
        
    bio5eco = numpy.array([])   # MaxTemperatureofWarmestMonth
    for t in range(NYears):
        b = numpy.mean(bio5[year == Years[t]])
        bio5eco = numpy.append(bio5eco, b)
        
    bio6eco = numpy.array([])  # MinTemperatureofColdestMonth
    for t in range(NYears):
        b = numpy.mean(bio6[year == Years[t]])
        bio6eco = numpy.append(bio6eco, b)
        
    bio7eco = numpy.array([])   # TemperatureAnnualRange
    for t in range(NYears):
        b = numpy.mean(bio7[year == Years[t]])
        bio7eco = numpy.append(bio7eco, b)
        
    bio8eco = numpy.array([])   # MeanTemperatureofWettestQuarter
    for t in range(NYears):
        b = numpy.mean(bio8[year == Years[t]])
        bio8eco = numpy.append(bio8eco, b)
        
    bio9eco = numpy.array([])  # MeanTemperatureofDriestQuarter
    for t in range(NYears):
        b = numpy.mean(bio9[year == Years[t]])
        bio9eco = numpy.append(bio9eco, b)
        
    bio10eco = numpy.array([])   # MeanTemperatureofWarmestQuarter
    for t in range(NYears):
        b = numpy.mean(bio10[year == Years[t]])
        bio10eco = numpy.append(bio10eco, b)
        
    bio11eco = numpy.array([])   # MeanTemperatureofColdestQuarter
    for t in range(NYears):
        b = numpy.mean(bio11[year == Years[t]])
        bio11eco = numpy.append(bio11eco, b)
        
    bio12eco = numpy.array([])   # AnnualPrecipitation
    for t in range(NYears):
        b = numpy.mean(bio12[year == Years[t]])
        bio12eco = numpy.append(bio12eco, b)
        
    bio13eco = numpy.array([])   # PrecipitationofWettestMonth
    for t in range(NYears):
        b = numpy.mean(bio13[year == Years[t]])
        bio13eco = numpy.append(bio13eco, b)
        
    bio14eco = numpy.array([])   # PrecipitationofDriestMonth
    for t in range(NYears):
        b = numpy.mean(bio14[year == Years[t]])
        bio14eco = numpy.append(bio14eco, b)
        
    bio15eco = numpy.array([])  # PrecipitationSeasonality
    for t in range(NYears):
        b = numpy.mean(bio15[year == Years[t]])
        bio15eco = numpy.append(bio15eco, b)
        
    bio16eco = numpy.array([])  # PrecipitationofWettestQuarter
    for t in range(NYears):
        b = numpy.mean(bio16[year == Years[t]])
        bio16eco = numpy.append(bio16eco, b)
        
    bio17eco = numpy.array([])   # PrecipitationofDriestQuarter
    for t in range(NYears):
        b = numpy.mean(bio17[year == Years[t]])
        bio17eco = numpy.append(bio17eco, b)
        
    bio18eco = numpy.array([])  # PrecipitationofWarmestQuarte
    for t in range(NYears):
        b = numpy.mean(bio18[year == Years[t]])
        bio18eco = numpy.append(bio18eco, b)
        
    bio19eco = numpy.array([])  # PrecipitationofColdestQuarter
    for t in range(NYears):
        b = numpy.mean(bio19[year == Years[t]])
        bio19eco = numpy.append(bio19eco, b)

    
   # create array - 1000 - copied climate vector to build regression
    # with Bayes simulated means:  
    bio1clim = numpy.tile(bio1eco,NSIMS)  
    bio2clim = numpy.tile(bio2eco,NSIMS) 
    bio3clim = numpy.tile(bio3eco,NSIMS) 
    bio4clim = numpy.tile(bio4eco,NSIMS) 
    bio5clim = numpy.tile(bio5eco,NSIMS)  
    bio6clim = numpy.tile(bio6eco,NSIMS) 
    bio7clim = numpy.tile(bio7eco,NSIMS) 
    bio8clim = numpy.tile(bio8eco,NSIMS) 
    bio9clim = numpy.tile(bio9eco,NSIMS) 
    bio10clim = numpy.tile(bio10eco,NSIMS) 
    bio11clim = numpy.tile(bio11eco,NSIMS) 
    bio12clim = numpy.tile(bio12eco,NSIMS) 
    bio13clim = numpy.tile(bio13eco,NSIMS) 
    bio14clim = numpy.tile(bio14eco,NSIMS) 
    bio15clim = numpy.tile(bio15eco,NSIMS) 
    bio16clim = numpy.tile(bio16eco,NSIMS) 
    bio17clim = numpy.tile(bio17eco,NSIMS) 
    bio18clim = numpy.tile(bio18eco,NSIMS) 
    bio19clim = numpy.tile(bio19eco,NSIMS) 
    

    return (bio1clim, bio2clim, bio3clim, bio4clim,
            bio5clim, bio6clim, bio7clim, bio8clim, 
            bio9clim, bio10clim, bio11clim, bio12clim, 
            bio13clim, bio14clim, bio15clim, bio16clim, 
            bio17clim, bio18clim, bio19clim)


###########################################################################
#########   Load data: ######################################
###########################################################################

# path = 'C:/Users/Olga Rumyantseva/Desktop/US_forest/' # home comp.
path = 'C:/Users/olga/Desktop/US_forest/'

# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df = df.drop('Unnamed: 0', 1)

# df.columns
df = df.dropna()

data = df.values 
NOBSERV = numpy.size(data, 0)

patch = data[:,21]   # Plot ID 
year = numpy.array([int(data[k,22]) for k in range(NOBSERV)]) # Year 
biomass = data[:,23] # Biomass
barea = data[:,24]   # Basal Area 

NSIMS = 1000

bio = getClimaticFactorsforRegress(data, NSIMS)


value = barea # this is what we consider now


Years = numpy.unique(year)
NYears = numpy.size(Years)

Patches  = numpy.unique(patch)
NPatches = numpy.size(Patches)


#   LogValuePatchYear[i][p] = log(biomass at plot p in year i):
LogValuePatchYear = [[0] * NPatches for i in range(NYears)]  
for k in range(NOBSERV):
    indY = numpy.where(Years == year[k])
    i = indY[0][0]     # return i such that year[k] == Years[i]
    indP = numpy.where(Patches == patch[k])
    j = indP[0][0]  # return j such that patch[k] == Patches[j]
    LogValuePatchYear[i][j] = math.log(value[k])
#  LogValuePatchYear[i - year][j - patch]


# means simulated by Bayes:
meansB =  [[0] * NSIMS for i in range(NYears)] 
#  meansB[year][simulation]

# vars simulated by Bayes:
varsB = [[0] * NSIMS for i in range(NYears)]
#  varsB[year][simulation]

# fill in the arrays meansB and varsB:
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
 


#  meansB[year][simulation]
BAsBayes = numpy.array([]) 

for n in range(NSIMS):
     for t in range(NYears):
          b = meansB[t][n]
          BAsBayes = numpy.append(BAsBayes, b)
    
numpy.size(BAsBayes)
    
###########################################################################
#########   Regression BA vs 1 clim var : ######################################
###########################################################################


dataset = pandas.DataFrame({'Basal Area':BAsBayes,
                            'AnnualMeanTemperature': bio[0],
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

