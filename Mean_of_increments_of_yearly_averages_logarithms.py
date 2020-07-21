# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:31:37 2020

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


path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'

df = pandas.read_csv(path + 'biodataUS_sorted.csv')

df.columns

df1 = df.iloc[:, [2, 3, 18, 19, 24, 25, 22, 23]]
df1.columns


data0 = df1.values  
NOBSERV0 = numpy.size(data0, 0) 

patch = data0[:,0]   # Plot ID 
year = numpy.array([int(data0[k,1]) for k in range(NOBSERV0)]) # Year 
biomass = data0[:,2] # Biomass
barea = data0[:,3]   # Basal Area
ecoreg = numpy.array([str(data0[k,4]) for k in range(NOBSERV0)])  # Eco region
state = numpy.array([str(data0[k,5]) for k in range(NOBSERV0)])   # US State



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

for eid in range(NEcoregions):
    patch = data0[:,0]   # Plot ID 
    year = numpy.array([int(data0[k,1]) for k in range(NOBSERV0)]) # Year 
    biomass = data0[:,2] # Biomass
    barea = data0[:,3]   # Basal Area
    ecoreg = numpy.array([str(data0[k,4]) for k in range(NOBSERV0)])  # Eco region
    state = numpy.array([str(data0[k,5]) for k in range(NOBSERV0)])   # US State
    df2 = df1[ecoreg == Ecoregions[eid]]
    data = df2.values  
    NOBSERV = numpy.size(data, 0) 

    patch = data[:,0]   # Plot ID 
    year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) # Year 
    barea = data[:,3]   # Basal Area
    ecoreg = numpy.array([str(data[k,4]) for k in range(NOBSERV)])  # Eco region
    value = barea # this is what we consider now: biomass 
    
    Years = numpy.unique(year)
    NYears = numpy.size(Years)
    
    YearAvBiomass = numpy.array([])
    for j in range(NYears):
        logbio = math.log(numpy.mean(value[year == Years[j]]))
        YearAvBiomass = numpy.append(YearAvBiomass, logbio)
        
    #Increments of average biomass (or shadow)
    DeltasYearAvBiomass = numpy.array([])  
    for i in range(numpy.size(YearAvBiomass)-1):
        delt = YearAvBiomass[i+1]-YearAvBiomass[i]
        DeltasYearAvBiomass = numpy.append(DeltasYearAvBiomass, delt)  
    
#    print(Ecoregions[eid], GrRate)
    print(round(numpy.mean(DeltasYearAvBiomass),3))

for eid in range(NEcoregions):
        print(Ecoregions[eid])