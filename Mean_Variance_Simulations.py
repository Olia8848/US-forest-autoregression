# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:25:16 2019

@author: olga
"""


import pandas   # data analysis
import numpy    # n-dim arrays
import math
import matplotlib
import random
import scipy    # lin regress
import matplotlib.pyplot as plt

from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import table





path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
df = pandas.read_csv(path + 'biodataUS_sorted.csv')
# df.columns
df1 = df.iloc[:, [2, 3, 18, 19, 24, 25, 22, 23]]
# df1.columns

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

eid = 1

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
    


mu = numpy.mean(DeltasYearAvBiomass)   # mean   numpy.mean(DeltasYearAvBiomass)
sigma = numpy.std(DeltasYearAvBiomass) # standard deviation   numpy.std(DeltasYearAvBiomass)

from math import e

# Years = numpy.unique(year)
# m0 = numpy.mean(value[year == Years[0]]  # Years[0] == 1970

#  means and variances based on existing data:
ExpectationsB = numpy.array([]) 
VariancesB = numpy.array([])
ExpectationsBA = numpy.array([]) 
VariancesBA = numpy.array([])



#  means and variances based on existing data:
for t in range(NYears): 
       meanB = numpy.mean(biomass[year == Years[t]])
       meanB = round(meanB, 2)
       ExpectationsB = numpy.append(ExpectationsB, meanB)
       
       vB = numpy.var(biomass[year == Years[t]])
       vB = round(math.sqrt(vB), 2)
       VariancesB = numpy.append(VariancesB, vB)
       
       meanBA = numpy.mean(barea[year == Years[t]])
       meanBA = round(meanBA, 2)
       ExpectationsBA = numpy.append(ExpectationsBA, meanBA)
       
       vBA = numpy.var(barea[year == Years[t]])
       vBA = round(math.sqrt(vBA), 2)
       VariancesBA = numpy.append(VariancesBA, vBA)
       
       nobserv = numpy.size(biomass[year == Years[t]])
       
       print(Years[t],', nobserv.:', nobserv, ', meanB:', meanB, ', vB:', vB,', meanBA:',  meanBA, ', vBA:', vBA, '\\')



#  simulated means and variances:
ExpB = numpy.array([]) 
VarB = numpy.array([])
ExpBA = numpy.array([]) 
VarBA = numpy.array([])



for t in range(2021 - Years[NYears-1]): 
       m0 = numpy.mean(biomass[year == Years[NYears-1]])
       meanB = round(m0*(e**(t*mu + t*(sigma**2)*(1/2))), 1)
       meanB = round(meanB, 2)
       ExpB = numpy.append(ExpB, meanB)
       
       vB = (m0**2)*(e**(2*t*mu + t*(sigma**2)*(1/2)))*(e**(t*(sigma**2)) - 1)
       vB = round(math.sqrt(vB), 2)
       VarB = numpy.append(VarB, vB)
       
       m0 = numpy.mean(barea[year == Years[0]])
       meanBA = round(m0*(e**(t*mu + t*(sigma**2)*(1/2))), 2)
       meanBA = round(meanBA, 2)
       
       ExpBA = numpy.append(ExpBA, meanBA)
       vBA = (m0**2)*(e**(2*t*mu + t*(sigma**2)*(1/2)))*(e**(t*(sigma**2)) - 1)
       vBA = round(math.sqrt(vBA), 1)
       VarBA = numpy.append(VarBA, vBA)
       
       print(Years[NYears-1]+t,', meanB:',  meanB, ', vB:', vB,', meanBA:',  meanBA, ', vBA:', vBA, '\\')
       


