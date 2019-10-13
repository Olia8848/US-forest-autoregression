# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:28:23 2019

@author: olga
"""

import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt


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
def qq(a, logvalue, lognext, interval, NCHANGES):
    V, A, r, sigma = regression(a, logvalue, lognext, interval, NCHANGES)
    centralized = numpy.array([V[k] - r*A[k] for k in range(NCHANGES)])
    qqplot(centralized, line = 'r') 
    pyplot.show()
    





path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
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


patch[0]   # Plot ID
year[0]    # Year
biomass[0] # Biomass
barea[0]   # Basal Area
ecoreg[0]  # Eco region
state[0]   # US State


Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions) # 36 ecoregions

States = numpy.unique(state)
NStates = numpy.size(States)  # 48 states

Years  = numpy.unique(year)
NYears = numpy.size(Years) # 40 years: 1968 - 2013
# 1968, 1970, 1972, 1974, 1977, 1978, 1980, 1981, 1982, 1983, 1984,
#       1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
#       1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
#      2007, 2008, 2009, 2010, 2011, 2012, 2013


###########################################################################
#########   Restricted by ecoregion: ######################################
###########################################################################

eid = 0

df2 = df1[ecoreg == Ecoregions[eid]]

data = df2.values  

NOBSERV = numpy.size(data, 0) 
print(NOBSERV)  # observations in your ecoregion


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
    

print('mean change of avg year biomass = ', numpy.mean(DeltasYearAvBiomass))
#Mean of increments of average yearly biomass
print('stdev of this change = ', numpy.std(DeltasYearAvBiomass))
#Standard deviation of these increments




Sigmas = numpy.array([])    
Rs = numpy.array([])

x = numpy.arange(-2, 2, 0.01)
for a in x:
    V, A, r, sigma = regression(a, logvalue, lognext, interval, NCHANGES)
    Sigmas = numpy.append(Sigmas, sigma)
    Rs = numpy.append(Rs, r)

#Plot of parameter a vs standard error
im = 'C:/Users/olga/Desktop/US_forest/pictures/'    
fig = plt.figure()
pyplot.plot(x, Sigmas)
pyplot.show()
fig.savefig(im + 'eco_'+ Ecoregions[eid] + '_Raw_Data_a_vs_stand_error.png')


r = Rs[numpy.argmin(Sigmas)]#Corresponding r

print('Non-centered data for all years')
print('min std = ', numpy.min(Sigmas))
print('value of a = ', x[numpy.argmin(Sigmas)])
print('value of m = ', r)
print('qq plot for residuals for random walk case')

qq(1, logvalue, lognext, interval, NCHANGES)
#residuals = [(lognext[k] - logvalue[k])/math.sqrt(interval[k]) - m * math.sqrt(interval[k]) for k in range(NCHANGES)]

    


