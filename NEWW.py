# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:57:23 2019

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


path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
df = pandas.read_csv(path + 'biodataUS_sorted.csv')
df.columns

df1 = df.iloc[:, [2, 3, 7, 8, 9, 18, 19, 24, 25, 22, 23]]
df1.columns

data0 = df1.values  
NOBSERV0 = numpy.size(data0, 0) 
print(NOBSERV0) #  number of observations totally 409 868



patch = data0[:,0]   # Plot ID 
year = numpy.array([int(data0[k,1]) for k in range(NOBSERV0)]) # Year 
SimpsonDB = data0[:,2]
sucessionDB = data0[:,3]
ShannonDB = data0[:,4]
biomass = data0[:,5] # Biomass
barea = data0[:,6]   # Basal Area
ecoreg = numpy.array([str(data0[k,7]) for k in range(NOBSERV0)])  # Eco region


uniqueValues, occurCount = numpy.unique(patch, return_counts=True)
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)

s = 0

N = numpy.size(occurCount)
for j in range(N):
     if (occurCount[j] > 8):
         s = s+1



numpy.corrcoef(SimpsonDB.astype(float), ShannonDB.astype(float), rowvar=False)

Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions) # 36 ecoregions


# document = Document()
# document.add_heading('correlations between Biomass and Basal Area for each ecoregion:')


###########################################################################
#########   Restricted by ecoregion: ######################################
###########################################################################
valuecorrs = numpy.array([])

for eid in range(NEcoregions):
     print(eid)

     df2 = df1[ecoreg == Ecoregions[eid]]
     data = df2.values  
     NOBSERV = numpy.size(data, 0) 
     
     biomass = data[:,2] # Biomass
     barea = data[:,3]   # Basal Area

     valuecorr = numpy.corrcoef(biomass.astype(float), barea.astype(float), rowvar=False)[0][1]
     valuecorrs = numpy.append(valuecorrs, round(valuecorr, 2))
#     document.add_paragraph('Ecoregion: ' + ' ' + str(Ecoregions[eid]) +  ', correlation: ' + str(valuecorr))
#     document.add_paragraph('  ')

path1 = 'C:/Users/olga/Desktop/US_forest/tables/'

res = pandas.DataFrame(valuecorrs)

res.to_csv(path1 + '_biom_barea_corr_coeff_table_2.csv')  

# document.save(path + 'Biomass_BArea_correlation_within_ecoregion.docx')





