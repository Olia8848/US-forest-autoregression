# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:44:38 2019

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


def Ev(a, n):
    if (a == -1) or (a == 1):
        return math.sqrt(n)
    else:
        b = ( 1-a**(2*n) ) / ( 1-a**2 )  
        return math.sqrt(b)

def Od(a, n): 
    if (a == 1):
        return n
    else:
        return (1-a**(n))/(1-a)   
     
# This is our regression with given value of a 
def regression(a, logvalue, lognext, interval, NCHANGES):
    V = numpy.array([])
    A = numpy.array([])
    for j in range(NCHANGES): 
         d = interval[j]
         diff = lognext[j]-(a**d)*logvalue[j]
         V = numpy.append(V, diff/Ev(a, d))  
         A = numpy.append(A, Od(a, d)/Ev(a, d))
    r = numpy.dot(V, A)/numpy.dot(A, A)
    sigma = math.sqrt((1/(NCHANGES-1))*numpy.dot(V-r*A, V-r*A))
    return V, A, r, sigma

# QQ plot of residuals for this regression
# compare to standard normal distribution, 'r' - regression line    
def qq(a, logvalue, lognext, interval, NCHANGES, im, eid):
    V, A, r, sigma = regression(a, logvalue, lognext, interval, NCHANGES)
    centralized = numpy.array([V[k] - r*A[k] for k in range(NCHANGES)])
    s1 = stats.shapiro(centralized)[0]
    s2 = stats.shapiro(centralized)[1]
    fig = qqplot(centralized, line = 'r') 
    pyplot.show()
    return s1, s2


path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
im = 'C:/Users/olga/Desktop/US_forest/pictures/'
df = pandas.read_csv(path + 'biodataUS_sorted.csv')
df.columns
df1 = df.iloc[:, [2, 3, 18, 19, 24, 25, 22, 23]]
df1.columns

data0 = df1.values  
NOBSERV0 = numpy.size(data0, 0) 
print(NOBSERV0) #  number of observations totally 409 868

patch = data0[:,0]   # Plot ID 
year = numpy.array([int(data0[k,1]) for k in range(NOBSERV0)]) # Year 
biomass = data0[:,2] # Biomass
barea = data0[:,3]   # Basal Area
ecoreg = numpy.array([str(data0[k,4]) for k in range(NOBSERV0)])  # Eco region
state = numpy.array([str(data0[k,5]) for k in range(NOBSERV0)])   # US State

Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions) # 36 ecoregions


###########################################################################
#########   Restricted by ecoregion: ######################################
###########################################################################
TestResults = numpy.array([])
TestResults1 = numpy.array([])

for eid in range(16, 35, 1):
    print(eid)
    
    ecoreg = numpy.array([str(data0[k,4]) for k in range(NOBSERV0)])  # Eco region
    df2 = df1[ecoreg == Ecoregions[eid]]
    data = df2.values  
    NOBSERV = numpy.size(data, 0) 
    
    patch = data[:,0]   # Plot ID 
    year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) # Year 
    biomass = data[:,2] # Biomass
    barea = data[:,3]   # Basal Area
    ecoreg = numpy.array([str(data[k,4]) for k in range(NOBSERV)])  # Eco region
    state = numpy.array([str(data[k,5]) for k in range(NOBSERV)])   # US State
    # value = biomass # this is what we consider now: biomass 
    value = biomass # this is what we consider now: biomass 
    
    logvalue = numpy.array([]) # here will be logarithms of value for steps
    lognext = numpy.array([])  # increments of logarithms 
    interval = numpy.array([]) # increments of years
    
    #next loop collects records from the same patch: 
    #initial , change in logs, time interval 
    #For example, if the same patch is observed in 1980 and 1987, 
    #collect logvalue in 1980, logchange and time interval
    
    for observ in range(NOBSERV-1):
        if patch[observ] == patch[observ+1]:
            logvalue = numpy.append(logvalue, math.log(value[observ]))
            lognext = numpy.append(lognext, math.log(value[observ+1]))
            interval = numpy.append(interval, year[observ+1] - year[observ])
    
           
    #Number of pair-observations
    NCHANGES = numpy.size(logvalue)  
    print('Number of pair-observations = ', NCHANGES)
    
    Years = numpy.unique(year)
    NYears = numpy.size(Years)
    
    #Average biomass or shadow in a given year
    YearAvBiomass = numpy.array([])
    for j in range(NYears):
        logbio = math.log(numpy.mean(value[year == Years[j]]))
        YearAvBiomass = numpy.append(YearAvBiomass, logbio)
        
    #Increments of average biomass (or shadow)
    DeltasYearAvBiomass = numpy.array([])  
    for i in range(numpy.size(YearAvBiomass)-1):
        delt = YearAvBiomass[i+1]-YearAvBiomass[i]
        DeltasYearAvBiomass = numpy.append(DeltasYearAvBiomass, delt)  
    
    Sigmas = numpy.array([])    
    Rs = numpy.array([])
    
    x = numpy.arange(-2, 2, 0.01)
    for a in x:
        V, A, r, sigma = regression(a, logvalue, lognext, interval, NCHANGES)
        Sigmas = numpy.append(Sigmas, sigma)
        Rs = numpy.append(Rs, r)
    
    f = qq(1, logvalue, lognext, interval, NCHANGES, im, eid)
    
    TestResults = numpy.append(TestResults, f[0])
    TestResults1 = numpy.append(TestResults1, f[1])
      

for i in range(19):
     print(round(TestResults[i], 2),'/', round(TestResults1[i], 2)) 