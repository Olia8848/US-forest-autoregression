# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:25:29 2020

@author: olga
"""

import pandas   
import numpy    
import math
import random
import matplotlib.pyplot as plt
import scipy

from scipy.stats import laplace
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
    s1 = stats.shapiro(centralized)[0]
    s2 = stats.shapiro(centralized)[1]
    qqplot(centralized, line = 'r') 
#    fig = plt.figure()
#    fig.savefig(im + 'eco_'+ Ecoregions[eid] + '_Raw_Data_residuals.png')
    pyplot.show()
    return s1, s2


def inverseGamma(alpha, beta):
    return (1/numpy.random.gamma(alpha, 1/beta))


   

# simulate Laplace transform sample
# loc = 0.0, scale = 1.0, size = 40
# loc = \mu - position of distribution peak. Default is 0.
# scale = \lambda - exponential decay. Default is 1
loc, scale = 0, 1

# number of observations totally 409 868
# observations in your ecoregion 14 060
# Npatches 8 095

Nplots = 8100  # number of unique forest plots
Nsteps = 60  # number of Laplace simulated BAs on each unique plot


# plot your sample histogram and Laplace curve together
# count, bins, ignored = plt.hist(barea1, 30, normed=True)
# x = numpy.arange(-8., 8., .01)
# pdf = numpy.exp(-abs(x-loc)/scale)/(2.*scale)
# plt.plot(x, pdf)
barea = numpy.array([])
patch = numpy.array([])
year = numpy.array([])

for i in range(Nplots):
    barea1 = numpy.random.laplace(loc, scale, size= Nsteps)  
# leave only every 5-th element in barea1 array
 # leave 5/40 = 12.5 % of laplace simulated data  
                         # Return evenly spaced values within a given interval
    ind = numpy.arange(0, barea1.size, 5)
    indlen = numpy.size(ind)
    barea1Select = barea1[ind]  
    barea = numpy.append(barea, barea1Select)
    year = numpy.append(year, ind)
    patch = numpy.append(patch, numpy.repeat(i, indlen))

patch = numpy.array([int(patch[k]) for k in range(numpy.size(patch))])
year = numpy.array([int(year[k]) for k in range(numpy.size(year))])

Years  = numpy.unique(year)
NYears = numpy.size(Years) 

logvalue = numpy.array([]) # here will be logarithms of value for steps
lognext = numpy.array([])  # increments of logarithms 
interval = numpy.array([]) # increments of years

NOBSERV = numpy.size(barea) 

for observ in range(NOBSERV-1):
    if patch[observ] == patch[observ+1]:
        logvalue = numpy.append(logvalue, barea[observ])
        lognext = numpy.append(lognext, barea[observ+1])
        interval = numpy.append(interval, year[observ+1] - year[observ])


#Number of pair-observations
NCHANGES = numpy.size(logvalue)  
print('Number of pair-observations = ', NCHANGES)




#Average biomass or shadow in a given year
YearAvBiomass = numpy.array([])
for j in range(NYears):
    logbio = numpy.mean(barea[year == Years[j]])
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


fig = plt.figure()
pyplot.plot(x, Sigmas)
pyplot.show()


r = Rs[numpy.argmin(Sigmas)]#Corresponding r

c1 = numpy.min(Sigmas)
c2 = x[numpy.argmin(Sigmas)]


print('Non-centered data for all years')
print('min std = ', c1)
print('value of a = ', c2)
print('value of m = ', r)


print('qq plot for residuals for random walk case')
qq(1, logvalue, lognext, interval, NCHANGES)

# null hypothesis - normality
# test statistic,  p-value (not normal if less than 0.05)  

YearAvBiomass

print(stats.shapiro(YearAvBiomass)[1])

qqplot(YearAvBiomass, line = 'r') 


