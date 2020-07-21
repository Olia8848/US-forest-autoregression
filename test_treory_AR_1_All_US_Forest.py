c# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas   # data analysis
import numpy    # n-dim arrays
import math
import matplotlib
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
import scipy    # lin regress
import random
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
data = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
data.columns
# precip. seasonality,
# mean temp of driest quart
# precip of driest month
# annual precip
# precip of coldest quarter
dataI = data.iloc[:, [22,23,24,25,26,27,1,2,17,11,16,14,21]]
dataI.columns
data0 = dataI.values  
NOBSERV0 = numpy.size(data, 0) 
# print(NOBSERV0) #  number of observations totally 409 868
patch = data0[:,0]   # Plot ID 
year = numpy.array([int(data0[k,1]) for k in range(NOBSERV0)]) # Year 
biomass = data0[:,2] # Biomass
barea = data0[:,3]   # Basal Area
shadeTol = data0[:,4]   # Basal Area
ecoreg = numpy.array([str(data0[k,5]) for k in range(NOBSERV0)])  # Eco region
clim1 = data0[:,8]   # PrecipitationSeasonality (all positives)
clim2 = data0[:,9]   # MeanTemperatureofDriestQuarter
clim3 = data0[:,10]   # PrecipitationofDriestMonth
clim4 = data0[:,11]   # AnnualPrecipitation  (all poitives)
clim5 = data0[:,12]   # PrecipitationofColdestQuarter (all poitives)

Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions) # 36 ecoregions

###########################################################################
#########   Restricted by ecoregion: ######################################
###########################################################################

BA = numpy.array([])
Cl1 = numpy.array([])
Cl4 = numpy.array([])
Cl5 = numpy.array([])

BAnext = numpy.array([])
BAprev = numpy.array([])
BAprev2 = numpy.array([])
BAprev3 = numpy.array([])
Cl1prev = numpy.array([])
Cl4prev = numpy.array([])
Cl5prev = numpy.array([])

for eid in range(NEcoregions):
    
    data = dataI[ecoreg == Ecoregions[eid]]
    data1 = data.values  
    NOBSERV = numpy.size(data1, 0) 
    # print(NOBSERV)  # observations in your ecoregion
    patch = data1[:,0]   # Plot ID 
    year = numpy.array([int(data1[k,1]) for k in range(NOBSERV)]) # Year 
    biomass = data1[:,2] # Biomass
    barea = data1[:,3]   # Basal Area
    shadeTol = data1[:,4]   # Basal Area
    clim1 = data1[:,8]   # PrecipitationSeasonality
    clim2 = data1[:,9]   # MeanTemperatureofDriestQuarter
    clim3 = data1[:,10]   # PrecipitationofDriestMonth
    clim4 = data1[:,11]   # AnnualPrecipitation
    clim5 = data1[:,12]   # AnnualPrecipitation
    
    Years = numpy.unique(year)
    NYears = numpy.size(Years)
    print(NYears-2)
    # Annual means :
    AnnualMeansBA = numpy.array([])
    AnnualMeansShTol = numpy.array([])
    AnnualMeansClim1 = numpy.array([])
    AnnualMeansClim4 = numpy.array([])
    AnnualMeansClim5 = numpy.array([])
    
    for i in range(NYears):
        logBA = numpy.mean(barea[year == Years[i]])
        AnnualMeansBA = numpy.append(AnnualMeansBA, logBA)
      
    for i in range(NYears):
        c1 = math.log(abs(numpy.mean(clim1[year == Years[i]])))
        AnnualMeansClim1 = numpy.append(AnnualMeansClim1, c1)
   
    for i in range(NYears):
        c4 = math.log(abs(numpy.mean(clim4[year == Years[i]])))
        AnnualMeansClim4 = numpy.append(AnnualMeansClim4, c4)
    
    for i in range(NYears):
        c5 = math.log(abs(numpy.mean(clim5[year == Years[i]])))
        AnnualMeansClim5 = numpy.append(AnnualMeansClim5, c5)    
    
    ############################################################
    DiffLnAnnMeansBA = numpy.array([])

    DiffLnAnnMeansCl1 = numpy.array([])
    DiffLnAnnMeansCl4 = numpy.array([])
    DiffLnAnnMeansCl5 = numpy.array([])
    
    DiffLnAnnMeansCl1prev = numpy.array([])
    DiffLnAnnMeansCl4prev = numpy.array([])
    DiffLnAnnMeansCl5prev = numpy.array([])    
    
    for i in range(numpy.size(AnnualMeansBA)-1):    
         d = AnnualMeansBA[i+1]-AnnualMeansBA[i]
         DiffLnAnnMeansBA = numpy.append(DiffLnAnnMeansBA, d)
         
    DiffLnAnnMeansBAnext = DiffLnAnnMeansBA[1:numpy.size(DiffLnAnnMeansBA)]
    DiffLnAnnMeansBAprev = DiffLnAnnMeansBA[0:(numpy.size(DiffLnAnnMeansBA)-1)]

         
    for i in range(numpy.size(AnnualMeansClim1)-1):    
         d = AnnualMeansClim1[i+1]-AnnualMeansClim1[i]
         DiffLnAnnMeansCl1 = numpy.append(DiffLnAnnMeansCl1, d)
   
    DiffLnAnnMeansCl1prev = DiffLnAnnMeansCl1[0:(numpy.size(DiffLnAnnMeansCl1)-1)]
         
    for i in range(numpy.size(AnnualMeansClim4)-1):    
         d = AnnualMeansClim4[i+1]-AnnualMeansClim4[i]
         DiffLnAnnMeansCl4 = numpy.append(DiffLnAnnMeansCl4, d)
    DiffLnAnnMeansCl4prev = DiffLnAnnMeansCl4[0:(numpy.size(DiffLnAnnMeansCl4)-1)]
     
    for i in range(numpy.size(AnnualMeansClim5)-1):    
         d = AnnualMeansClim5[i+1]-AnnualMeansClim5[i]
         DiffLnAnnMeansCl5 = numpy.append(DiffLnAnnMeansCl5, d)     
    DiffLnAnnMeansCl5prev = DiffLnAnnMeansCl5[0:(numpy.size(DiffLnAnnMeansCl5)-1)]
     
    BA = numpy.append(BA, DiffLnAnnMeansBA)
    BAnext = numpy.append(BAnext, DiffLnAnnMeansBAnext)
    BAprev = numpy.append(BAprev, DiffLnAnnMeansBAprev)
    BAprev2 = numpy.append(BAprev2, DiffLnAnnMeansBAprev)
    BAprev3 = numpy.append(BAprev3, DiffLnAnnMeansBAprev)

    Cl1 = numpy.append(Cl1, DiffLnAnnMeansCl1)
    Cl4 = numpy.append(Cl4, DiffLnAnnMeansCl4)
    Cl5 = numpy.append(Cl5, DiffLnAnnMeansCl5)
    Cl1prev = numpy.append(Cl1prev, DiffLnAnnMeansCl1prev)
    Cl4prev = numpy.append(Cl4prev, DiffLnAnnMeansCl4prev)
    Cl5prev = numpy.append(Cl5prev, DiffLnAnnMeansCl5prev)



UnitsVec = numpy.repeat(1, numpy.size(BAprev))



import statsmodels.api as sm
import ols

X = pandas.DataFrame({'BA': BA,
                      'BAprev': BAprev,
                      'BAprev2': BAprev2,
                      'BAprev3': BAprev3,
                      'BAnext': BAnext,
                      'Cl1prev': Cl1prev,
                      'Cl4prev': Cl4prev,
                      'Cl5prev': Cl5prev,
                      'Cl1': Cl1,
                      'Cl4': Cl4,
                      'Cl5': Cl5,})
 
reg = ols.ols(formula = 'BAnext ~ BAprev', data=X).fit()
print (reg.summary())

plt.scatter(BAnext, BAprev)


    