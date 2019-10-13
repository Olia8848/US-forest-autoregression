# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:06:03 2019

@author: olga
"""


import pandas   # data analysis
import numpy    # n-dim arrays
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import scipy    # lin regress
import matplotlib.pyplot as plt
#from pandas.tools.plotting import table
import random




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
value = barea # this is what we consider now: biomass 



##################################################################
#####  Annual means of biomass and shadow: ########################
##################################################################


Years = numpy.unique(year)
NYears = numpy.size(Years)

# Annual means of biomass:
AnnualMeanBiomass = numpy.array([]) 

for i in range(NYears):
    logbio = math.log(numpy.mean(value[year == Years[i]]))
    AnnualMeanBiomass = numpy.append(AnnualMeanBiomass, logbio)
    

# Annual means of shadows:
AnnualMeanBA = numpy.array([]) 

for i in range(NYears):
    logBA = math.log(numpy.mean(barea[year == Years[i]]))
    AnnualMeanBA = numpy.append(AnnualMeanBA, logBA)
    

##################################################################
#####  Plots of annual means of biomass and shadow: ###############
##################################################################
im = 'C:/Users/olga/Desktop/US_forest/pictures/'    

fig1 = plt.figure()
pyplot.plot(Years, AnnualMeanBiomass, 'bo', Years, AnnualMeanBiomass, 'k')
pyplot.show()
fig1.savefig(im + 'eco_' + Ecoregions[eid] + '_AnnualMeanBiomass.png')


fig2 = plt.figure()
pyplot.plot(Years, AnnualMeanBA, 'bo', Years, AnnualMeanBA, 'k')
pyplot.show()
fig2.savefig(im + 'eco_'+ Ecoregions[eid] + '_AnnualMeanBA.png')


########################  Table (38*4)  ###########################

# Year|| 
# of observ ||
# Mean of biom. this year || 
# Mean of shadow this year  ||

####################################################################

NObserv = [0] * NYears 

for i in range(NYears):
    NObserv[i] = numpy.size(value[year == Years[i]])

dataT3 = {'Year':Years,'Number of Observations':NObserv, 
'mean of biomass':AnnualMeanBiomass,'mean of basal area':AnnualMeanBA}
df3 = pandas.DataFrame(dataT3)
print(df3)

df3.to_csv(path + 'eco_'+ Ecoregions[eid] + '_means_biomass_BA.csv', index=False)


##################################################################
#####  Distribution of patches observed in 2007: ###############
##################################################################

# the last year for which we have observations is Years[NYears-1]

PatchesLastObsYear = patch[year == Years[NYears-1]]
BiomassLastObsYear = biomass[year == Years[NYears-1]]
BALastObsYear = barea[year == Years[NYears-1]]


# pyplot.hist(biomass[year == 2012] , bins = 100)
# pyplot.show()

# pyplot.hist(BALastObsYear, bins = 100)
# pyplot.show()

##################################################################
#####  Simulate Random Walk 12 years ahead from 2007
# for some patch which was observed in 2007: ###############
##################################################################


logvalue = numpy.array([]) # here will be logarithms of value for steps
lognext = numpy.array([])  # increments of logarithms 
interval = numpy.array([]) # increments of years


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


mu = numpy.mean(DeltasYearAvBiomass)   
sigma = numpy.std(DeltasYearAvBiomass) 

Iteartions = 1000  # the number of simulations 


# for ex., 1000 simulations of 12-year forward predictions:
# select randomly patch which was observed in 2007
# (Y0,Y2, ... , Y11) ... (Y0,Y2, ... , Y11) - 1000 arrays like this
# Y0 - in 2008 (1st prediction), ... , Y11 - in 2019

n = 2022 - Years[NYears-1]
 
Simulations = [[0] * n for i in range(Iteartions)] 

PatchesLastObsYear = patch[year == Years[NYears-1]]

for i in range(Iteartions):
    dat1 = data[(year == Years[NYears-1]) & (patch == random.choice(PatchesLastObsYear))]
    value = dat1[0][2] # biomass
    Simulations[i][0] = value + numpy.random.normal(mu, sigma) # 2008 prediction
#   Switch between these two:
#   value = dat1[0][2] # biomass on the chosen patch in 2007
#   value = dat1[0][3] # bas. area on the chosen patch in 2007

Simulations[100][0]

for j in range(n-1):
    print('year =', Years[NYears-1]+j+2) # doing simulation for this year
    for i in range(Iteartions):
        Simulations[i][j+1] = Simulations[i][j] + numpy.random.normal(mu, sigma) 
    

    
Predictions = numpy.array([]) 

for j in range(n):
    sim = numpy.array([])
    for i in range(Iteartions):
        sim = numpy.append(sim, Simulations[i][j])
    Predictions = numpy.append(Predictions, math.log(numpy.mean(sim)))


YearsPred = numpy.array([])

for j in range(n):
    YearsPred = numpy.append(YearsPred, Years[NYears-1] + j+1)


fig, ax = pyplot.subplots()
pyplot.plot(YearsPred, Predictions, 'bo', YearsPred, Predictions, 'k', color='tab:green')
ax.ticklabel_format(useOffset=False)
pyplot.plot(Years, AnnualMeanBiomass, 'bo', Years, AnnualMeanBiomass, 'k')
plt.show()








