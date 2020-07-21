# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 06:38:49 2019

@author: olga
"""

import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import mean
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy import stats

path = 'C:/Users/olga/Desktop/US_forest/'
df = pandas.read_csv(path + 'biodataUS_sorted.csv')
df1 = df.iloc[:, [2, 3, 18, 19, 24, 25, 22, 23]]
data0 = df1.values  
NOBSERV0 = numpy.size(data0, 0) 
ecoreg = numpy.array([str(data0[k,4]) for k in range(NOBSERV0)])  # Eco region
Ecoregions  = numpy.unique(ecoreg)

    
path1 = 'C:/Users/olga/Desktop/US_forest/tables/'

###################################################################################
#################  Biomass:  ###################################################
###################################################################################

MB = numpy.genfromtxt(path1 + 'CrossCorrelations_biomass_for_load.csv', delimiter=',', names=True)

A = [[0] * 36 for i in range(36)] 

for i in range(36):
    for j in range(36):
        A[i][j] = MB[i][j]
#        if ((A[i][j] > 10) & (i<=j)):
#            print(MB[i][j])
#            print(i, j, Ecoregions[i], Ecoregions[j], MB[i][j])
        

# print maximal positive cross-correlations:
for j in range(36):
    vec = numpy.array([])
    for i in range(36):
        vec = numpy.append(vec, A[i][j])
#    print(j, numpy.nanargmax(vec), Ecoregions[j], Ecoregions[numpy.nanargmax(vec)], numpy.nanmax(vec))  
    print(numpy.nanmax(vec))
    
    
# print minimal negative cross-correlations:
for j in range(36):
    vec = numpy.array([])
    for i in range(36):
        vec = numpy.append(vec, A[i][j])
#    print(j, numpy.nanargmin(vec), Ecoregions[j], Ecoregions[numpy.nanargmin(vec)], numpy.nanmin(vec))  
    print(numpy.nanmin(vec))
        
        

mask = numpy.zeros_like(A)
mask[numpy.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = plt.axes()
    plt.title('Biomass Cross-Correlations', fontsize = 20) 
    ax = sns.heatmap(A, mask=mask, square=True,  cmap="BuPu")
    plt.xlabel('ecoregion', fontsize = 15) 
    plt.ylabel('ecoregion', fontsize = 15)
    plt.show()

###################################################################################
#################  Basal Area:  ###################################################
###################################################################################

MBar = numpy.genfromtxt(path1 + 'CrossCorrelations_BArea_for_load.csv', delimiter=',', names=True)

B = [[0] * 36 for i in range(36)] 

for i in range(36):
    for j in range(36):
        B[i][j] = MBar[i][j]
#        if ((B[i][j] > 10) & (i<=j)):
#            print(MBar[i][j])
#            print(i, j, Ecoregions[i], Ecoregions[j], MBar[i][j])
        


# print maximal positive cross-correlations:
for j in range(36):
    vec = numpy.array([])
    for i in range(36):
        vec = numpy.append(vec, B[i][j])
#    print(j, numpy.nanargmax(vec), Ecoregions[j], Ecoregions[numpy.nanargmax(vec)], numpy.nanmax(vec))  
    print(numpy.nanmax(vec))
    
    
# print minimal negative cross-correlations:
for j in range(36):
    vec = numpy.array([])
    for i in range(36):
        vec = numpy.append(vec, B[i][j])
#    print(j, numpy.nanargmin(vec), Ecoregions[j], Ecoregions[numpy.nanargmin(vec)], numpy.nanmin(vec))  
    print(numpy.nanmin(vec))

    

mask = numpy.zeros_like(B)
mask[numpy.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = plt.axes()
    plt.title('Basal Area Cross-Correlations', fontsize = 20) 
    ax = sns.heatmap(A, mask=mask, square=True,  cmap="BuPu")
    plt.xlabel('ecoregion', fontsize = 15) 
    plt.ylabel('ecoregion', fontsize = 15)
    plt.show()
