# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:32:34 2019

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
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

path = 'C:/Users/olga/Desktop/US_forest/'

# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'biodataUS_sorted_climatic.csv')
df = df.drop('Unnamed: 0', 1)
df = df.dropna()
data = df.values  
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

biomass = data[:,23] # Biomass
barea = data[:,24]   # Basal Area

value = barea

dataset = pandas.DataFrame({'Basal Area':value,
                            'AnnualMeanTemperature': bio1,
                            'MeanDiurnalRange': bio2,
                            'Isothermality': bio3,
                            'TemperatureSeasonality': bio4,
                            'MaxTemperatureofWarmestMonth': bio5,
                            'MinTemperatureofColdestMonth': bio6,
                            'TemperatureAnnualRange': bio7,
                            'MeanTemperatureofWettestQuarter': bio8,
                            'MeanTemperatureofDriestQuarter': bio9,
                            'MeanTemperatureofWarmestQuarter': bio10,
                            'MeanTemperatureofColdestQuarter': bio11,
                            'AnnualPrecipitation': bio12,
                            'PrecipitationofWettestMonth': bio13,
                            'PrecipitationofDriestMonth': bio14,
                            'PrecipitationSeasonality': bio15,
                            'PrecipitationofWettestQuarter': bio16,
                            'PrecipitationofDriestQuarter': bio17,
                            'PrecipitationofWarmestQuarte': bio18,
                            'PrecipitationofColdestQuarter': bio19})
    
dataset.to_csv(path + 'BA_clim_characts_df_for_PCA.csv')
