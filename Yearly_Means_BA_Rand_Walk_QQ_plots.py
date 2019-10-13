# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:52:55 2019

@author: olga
"""

import pandas   
import numpy    
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import scipy
from scipy import stats


path = 'C:/Users/olga/Desktop/US_forest/'
im = 'C:/Users/olga/Desktop/US_forest/QQ_plots_yearly_means/'

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

Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions) # 36 ecoregions



MeanResults = numpy.array([])
StdResults = numpy.array([])
TestResults = numpy.array([])
TestResults1 = numpy.array([])


for eid in range(36):

    df2 = df1[ecoreg == Ecoregions[eid]]
    data = df2.values  
    NOBSERV = numpy.size(data, 0) 
    patch = data[:,0]   # Plot ID 
    year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) # Year 
    biomass = data[:,2] # Biomass
    barea = data[:,3]   # Basal Area
    value = biomass 
    Years = numpy.unique(year)
    NYears = numpy.size(Years)
    
    # Average biomass or shadow in a given year 
    YearAvBiomass = numpy.array([])
    for j in range(NYears):
        logbio = math.log(numpy.mean(value[year == Years[j]]))
        YearAvBiomass = numpy.append(YearAvBiomass, logbio)    
    #Increments of average biomass (or shadow)
    DeltasYearAvBiomass = numpy.array([])  
    for i in range(numpy.size(YearAvBiomass)-1):
        delt = YearAvBiomass[i+1]-YearAvBiomass[i]
        DeltasYearAvBiomass = numpy.append(DeltasYearAvBiomass, delt)  
    
    MeanDeltasAvg = numpy.mean(DeltasYearAvBiomass)
    StdDeltasAvg = numpy.std(DeltasYearAvBiomass)
    
    print(MeanDeltasAvg)
    print(StdDeltasAvg)
    print(stats.shapiro(DeltasYearAvBiomass)[0])
    print(stats.shapiro(DeltasYearAvBiomass)[1])
    
    MeanResults = numpy.append(MeanResults, MeanDeltasAvg)
    StdResults = numpy.append(StdResults, StdDeltasAvg)
    TestResults = numpy.append(TestResults, stats.shapiro(DeltasYearAvBiomass)[0])
    TestResults1 = numpy.append(TestResults1, stats.shapiro(DeltasYearAvBiomass)[1])
  
    print(eid)
    
    print('QQ plot for changes in yearly average')
    qqplot(DeltasYearAvBiomass, line = 'r')
    pyplot.show()



for i in range(36):
     print(round(MeanResults[i], 2),'/', round(StdResults[i], 2)) 



for i in range(36):
     print(round(TestResults[i], 2),'/', round(TestResults1[i], 2)) 


# This shows that these yearly changes obey normal distributions
# The test statistic 0.9816, 
# p-value 0.7867 - big, so we can't reject the
# null-hypothesis => normally distributed

