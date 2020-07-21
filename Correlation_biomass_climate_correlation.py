# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:06:40 2019

@author: olga
"""

# Annual Mean Temperature (BIO1)
# Mean Diurnal Range (BIO2)  
# Isothermality (BIO3)
# Temperature Seasonality (BIO4)
# Max Temperature of Warmest Month (BIO5)     
# Min Temperature of Coldest Month (BIO6)   
# Temperature Annual Range (BIO7)
# Mean Temperature of Wettest Quarter (BIO8)
# Mean Temperature of Driest Quarter (BIO9)   
# Mean Temperature of Warmest Quarter (BIO10) 
# Mean Temperature of Coldest Quarter (BIO11) 

# Annual Precipitation (BIO12) 
# Precipitation of Wettest Month (BIO13)
# Precipitation of Driest Month (BIO14)   
# Precipitation Seasonality (BIO15)
# Precipitation of Wettest Quarter (BIO16) 
# Precipitation of Driest Quarter (BIO17)   
# Precipitation of Warmest Quarter (BIO18)  
# Precipitation of Coldest Quarter (BIO19)


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



path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'

# in the data below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df = df.dropna()


data0 = df.values  
NOBSERV0 = numpy.size(data0, 0) 
ecoreg = numpy.array([str(data0[k,25]) for k in range(NOBSERV0)])  # Eco region
Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions)

df = df.drop('Unnamed: 0', 1)
df.columns

########################################################################
###############  corr for the whole data: #############################
######################################################################

bio1 = data0[:,2]  # AnnualMeanTemperature
bio2 = data0[:,3]  # MeanDiurnalRange
bio3 = data0[:,4]  # Isothermality
bio4 = data0[:,5]  # TemperatureSeasonality
bio5 = data0[:,6]  # MaxTemperatureofWarmestMonth
bio6 = data0[:,7]  # MinTemperatureofColdestMonth
bio7 = data0[:,8]  # TemperatureAnnualRange
bio8 = data0[:,9]  # MeanTemperatureofWettestQuarter
bio9 = data0[:,10]  # MeanTemperatureofDriestQuarter
bio10 = data0[:,11]  # MeanTemperatureofWarmestQuarter
bio11 = data0[:,12]  # MeanTemperatureofColdestQuarter
bio12 = data0[:,13]  # AnnualPrecipitation
bio13 = data0[:,14]  # PrecipitationofWettestMonth
bio14 = data0[:,15]  # PrecipitationofDriestMonth
bio15 = data0[:,16]  # PrecipitationSeasonality
bio16 = data0[:,17]  # PrecipitationofWettestQuarter
bio17 = data0[:,18]  # PrecipitationofDriestQuarter
bio18 = data0[:,19]  # PrecipitationofWarmestQuarte
bio19 = data0[:,20]  # PrecipitationofColdestQuarter

biomass = data0[:,23] # Biomass
barea = data0[:,24]   # Basal Area

value = biomass # this is what we consider now: biomass 

c1 = numpy.corrcoef(value.astype(float), bio1.astype(float), rowvar=False)[0][1]
c2 = numpy.corrcoef(value.astype(float), bio2.astype(float), rowvar=False)[0][1]
c3 = numpy.corrcoef(value.astype(float), bio3.astype(float), rowvar=False)[0][1]
c4 = numpy.corrcoef(value.astype(float), bio4.astype(float), rowvar=False)[0][1]
c5 = numpy.corrcoef(value.astype(float), bio5.astype(float), rowvar=False)[0][1]
c6 = numpy.corrcoef(value.astype(float), bio6.astype(float), rowvar=False)[0][1]
c7 = numpy.corrcoef(value.astype(float), bio7.astype(float), rowvar=False)[0][1]
c8 = numpy.corrcoef(value.astype(float), bio8.astype(float), rowvar=False)[0][1]
c9 = numpy.corrcoef(value.astype(float), bio9.astype(float), rowvar=False)[0][1]
c10 = numpy.corrcoef(value.astype(float), bio10.astype(float), rowvar=False)[0][1]
c11 = numpy.corrcoef(value.astype(float), bio11.astype(float), rowvar=False)[0][1]
c12 = numpy.corrcoef(value.astype(float), bio12.astype(float), rowvar=False)[0][1]
c13 = numpy.corrcoef(value.astype(float), bio13.astype(float), rowvar=False)[0][1]
c14 = numpy.corrcoef(value.astype(float), bio14.astype(float), rowvar=False)[0][1]
c15 = numpy.corrcoef(value.astype(float), bio15.astype(float), rowvar=False)[0][1]
c16 = numpy.corrcoef(value.astype(float), bio16.astype(float), rowvar=False)[0][1]
c17 = numpy.corrcoef(value.astype(float), bio17.astype(float), rowvar=False)[0][1]
c18 = numpy.corrcoef(value.astype(float), bio18.astype(float), rowvar=False)[0][1]
c19 = numpy.corrcoef(value.astype(float), bio19.astype(float), rowvar=False)[0][1]

value = barea # this is what we consider now: biomass 

d1 = numpy.corrcoef(value.astype(float), bio1.astype(float), rowvar=False)[0][1]
d2 = numpy.corrcoef(value.astype(float), bio2.astype(float), rowvar=False)[0][1]
d3 = numpy.corrcoef(value.astype(float), bio3.astype(float), rowvar=False)[0][1]
d4 = numpy.corrcoef(value.astype(float), bio4.astype(float), rowvar=False)[0][1]
d5 = numpy.corrcoef(value.astype(float), bio5.astype(float), rowvar=False)[0][1]
d6 = numpy.corrcoef(value.astype(float), bio6.astype(float), rowvar=False)[0][1]
d7 = numpy.corrcoef(value.astype(float), bio7.astype(float), rowvar=False)[0][1]
d8 = numpy.corrcoef(value.astype(float), bio8.astype(float), rowvar=False)[0][1]
d9 = numpy.corrcoef(value.astype(float), bio9.astype(float), rowvar=False)[0][1]
d10 = numpy.corrcoef(value.astype(float), bio10.astype(float), rowvar=False)[0][1]
d11 = numpy.corrcoef(value.astype(float), bio11.astype(float), rowvar=False)[0][1]
d12 = numpy.corrcoef(value.astype(float), bio12.astype(float), rowvar=False)[0][1]
d13 = numpy.corrcoef(value.astype(float), bio13.astype(float), rowvar=False)[0][1]
d14 = numpy.corrcoef(value.astype(float), bio14.astype(float), rowvar=False)[0][1]
d15 = numpy.corrcoef(value.astype(float), bio15.astype(float), rowvar=False)[0][1]
d16 = numpy.corrcoef(value.astype(float), bio16.astype(float), rowvar=False)[0][1]
d17 = numpy.corrcoef(value.astype(float), bio17.astype(float), rowvar=False)[0][1]
d18 = numpy.corrcoef(value.astype(float), bio18.astype(float), rowvar=False)[0][1]
d19 = numpy.corrcoef(value.astype(float), bio19.astype(float), rowvar=False)[0][1]


climcorrbio = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19] 
climcorrba =  [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19] 


###########################################################################
#########   corr. for each ecoregion: ######################################
###########################################################################

#document = Document()
#document.add_heading('Correlations between climate and tree biomass:')

NOBSERVs = numpy.array([]) 
Yearss = numpy.array([])
Statess = numpy.array([])
NYearss = numpy.array([]) 

bImpBiomasss = numpy.array([]) # highest positive corr
bImpBas = numpy.array([])      # highest positive corr
bNImpBiomasss = numpy.array([]) # highest negative corr
bNImpBas = numpy.array([])    # highest negative corr   



for eid in range(NEcoregions):
   # j = 0
    data0 = df.values  
    NOBSERV0 = numpy.size(data0, 0)    
    ecoreg = numpy.array([str(data0[k,25]) for k in range(NOBSERV0)])  # Eco region
    Ecoregions  = numpy.unique(ecoreg)
    
    df2 = df[ecoreg == Ecoregions[eid]]
    data = df2.values
    NOBSERV = numpy.size(data, 0)
    NOBSERVs = numpy.append(NOBSERVs, NOBSERV)
     # inside the ecoregion:   
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
    Yearss = numpy.append(Yearss, Years, axis=0)
    
    NYears = numpy.size(Years)
    NYearss = numpy.append(NYearss, NYears)


    state = numpy.array([str(data[k,26]) for k in range(NOBSERV)])   # US State
    States = numpy.unique(state)
    Statess = numpy.append(Statess, States, axis=0)
    
    NStates = numpy.size(States)
    
    biomass = data[:,23] # Biomass
    barea = data[:,24]   # Basal Area

    value = biomass # this is what we consider now: biomass 
    numpy.size(bio1)
    
    c1 = numpy.corrcoef(value.astype(float), bio1.astype(float), rowvar=False)[0][1]
    c2 = numpy.corrcoef(value.astype(float), bio2.astype(float), rowvar=False)[0][1]
    c3 = numpy.corrcoef(value.astype(float), bio3.astype(float), rowvar=False)[0][1]
    c4 = numpy.corrcoef(value.astype(float), bio4.astype(float), rowvar=False)[0][1]
    c5 = numpy.corrcoef(value.astype(float), bio5.astype(float), rowvar=False)[0][1]
    c6 = numpy.corrcoef(value.astype(float), bio6.astype(float), rowvar=False)[0][1]
    c7 = numpy.corrcoef(value.astype(float), bio7.astype(float), rowvar=False)[0][1]
    c8 = numpy.corrcoef(value.astype(float), bio8.astype(float), rowvar=False)[0][1]
    c9 = numpy.corrcoef(value.astype(float), bio9.astype(float), rowvar=False)[0][1]
    c10 = numpy.corrcoef(value.astype(float), bio10.astype(float), rowvar=False)[0][1]
    c11 = numpy.corrcoef(value.astype(float), bio11.astype(float), rowvar=False)[0][1]
    c12 = numpy.corrcoef(value.astype(float), bio12.astype(float), rowvar=False)[0][1]
    c13 = numpy.corrcoef(value.astype(float), bio13.astype(float), rowvar=False)[0][1]
    c14 = numpy.corrcoef(value.astype(float), bio14.astype(float), rowvar=False)[0][1]
    c15 = numpy.corrcoef(value.astype(float), bio15.astype(float), rowvar=False)[0][1]
    c16 = numpy.corrcoef(value.astype(float), bio16.astype(float), rowvar=False)[0][1]
    c17 = numpy.corrcoef(value.astype(float), bio17.astype(float), rowvar=False)[0][1]
    c18 = numpy.corrcoef(value.astype(float), bio18.astype(float), rowvar=False)[0][1]
    c19 = numpy.corrcoef(value.astype(float), bio19.astype(float), rowvar=False)[0][1]
    
    value = barea # this is what we consider now: biomass 
    
    d1 = numpy.corrcoef(value.astype(float), bio1.astype(float), rowvar=False)[0][1]
    d2 = numpy.corrcoef(value.astype(float), bio2.astype(float), rowvar=False)[0][1]
    d3 = numpy.corrcoef(value.astype(float), bio3.astype(float), rowvar=False)[0][1]
    d4 = numpy.corrcoef(value.astype(float), bio4.astype(float), rowvar=False)[0][1]
    d5 = numpy.corrcoef(value.astype(float), bio5.astype(float), rowvar=False)[0][1]
    d6 = numpy.corrcoef(value.astype(float), bio6.astype(float), rowvar=False)[0][1]
    d7 = numpy.corrcoef(value.astype(float), bio7.astype(float), rowvar=False)[0][1]
    d8 = numpy.corrcoef(value.astype(float), bio8.astype(float), rowvar=False)[0][1]
    d9 = numpy.corrcoef(value.astype(float), bio9.astype(float), rowvar=False)[0][1]
    d10 = numpy.corrcoef(value.astype(float), bio10.astype(float), rowvar=False)[0][1]
    d11 = numpy.corrcoef(value.astype(float), bio11.astype(float), rowvar=False)[0][1]
    d12 = numpy.corrcoef(value.astype(float), bio12.astype(float), rowvar=False)[0][1]
    d13 = numpy.corrcoef(value.astype(float), bio13.astype(float), rowvar=False)[0][1]
    d14 = numpy.corrcoef(value.astype(float), bio14.astype(float), rowvar=False)[0][1]
    d15 = numpy.corrcoef(value.astype(float), bio15.astype(float), rowvar=False)[0][1]
    d16 = numpy.corrcoef(value.astype(float), bio16.astype(float), rowvar=False)[0][1]
    d17 = numpy.corrcoef(value.astype(float), bio17.astype(float), rowvar=False)[0][1]
    d18 = numpy.corrcoef(value.astype(float), bio18.astype(float), rowvar=False)[0][1]
    d19 = numpy.corrcoef(value.astype(float), bio19.astype(float), rowvar=False)[0][1]
    
    
    climcorrbio = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19] 
    climcorrba =  [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19] 
    bImpBiomass = numpy.argmax(climcorrbio)
    bImpBa = numpy.argmax(climcorrba)
    
    bNImpBiomass = numpy.argmin(climcorrbio)
    bNImpBa = numpy.argmin(climcorrba)
    
    print(eid, numpy.max(climcorrba), numpy.max(climcorrbio), numpy.min(climcorrba), numpy.min(climcorrbio))
    
    bImpBiomasss = numpy.append(bImpBiomasss, bImpBiomass+1)
    bImpBas = numpy.append(bImpBas, bImpBa+1)
    bNImpBiomasss = numpy.append(bNImpBiomasss, bNImpBiomass+1)
    bNImpBas = numpy.append(bNImpBas, bNImpBa+1)
    
    print(Ecoregions[eid], NYears, Years, States)
    print(" ")


BioVarsAllNames = ['Annual Mean Temperature (BIO1)',
'Mean Diurnal Range (BIO2)',  
'Isothermality (BIO3)',
'Temperature Seasonality (BIO4)',
'Max Temperature of Warmest Month (BIO5)',     
'Min Temperature of Coldest Month (BIO6)',   
'Temperature Annual Range (BIO7)',
'Mean Temperature of Wettest Quarter (BIO8)',
'Mean Temperature of Driest Quarter (BIO9)',   
'Mean Temperature of Warmest Quarter (BIO10)', 
'Mean Temperature of Coldest Quarter (BIO11)', 
'Annual Precipitation (BIO12)', 
'Precipitation of Wettest Month (BIO13)',
'Precipitation of Driest Month (BIO14)',   
'Precipitation Seasonality (BIO15)',
'Precipitation of Wettest Quarter (BIO16)', 
'Precipitation of Driest Quarter (BIO17)',   
'Precipitation of Warmest Quarter (BIO18)',  
'Precipitation of Coldest Quarter (BIO19)']

# BiomMaxCorrs = numpy.array([]) # highest positive corr
# BiomMinCorrs = numpy.array([])      # highest positive corr
# BAMaxCorrs = numpy.array([]) # highest negative corr
# BAMinCorrs = numpy.array([])    # highest negative corr   


# for i in range(36):
#    BiomMaxCorrs = numpy.append(BiomMaxCorrs, BioVarsAllNames[int(bImpBiomasss[i])-1])
#    BiomMinCorrs = numpy.append(BiomMinCorrs, BioVarsAllNames[int(bNImpBiomasss[i])-1])
#    BAMaxCorrs = numpy.append(BAMaxCorrs, BioVarsAllNames[int(bImpBas[i])-1])
#    BAMinCorrs = numpy.append(BAMinCorrs, BioVarsAllNames[int(bNImpBas[i])-1])

path1 = 'C:/Users/olga/Desktop/US_forest/tables/'

dat = pandas.concat([pandas.DataFrame(Ecoregions), 
                     pandas.DataFrame(NOBSERVs.astype(int)), 
                     pandas.DataFrame(bImpBiomasss.astype(int)), 
                     pandas.DataFrame(bNImpBiomasss.astype(int)), 
                     pandas.DataFrame(bImpBas.astype(int)), 
                     pandas.DataFrame(bNImpBas.astype(int))], axis = 1)

res = pandas.DataFrame(dat)

res.to_csv(path1 + 'table_1.csv')


