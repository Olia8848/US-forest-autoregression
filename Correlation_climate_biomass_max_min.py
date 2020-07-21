# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:09:13 2019

@author: olga
"""

import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt
import random
import docx

from numpy import mean
from matplotlib import pyplot

path = 'C:/Users/olga/Desktop/US_forest/tables/'

# in the dataset below NA-s in LAT and LON were removed:
df = pandas.read_csv(path + 'table_1.csv')
df.columns

data = df.values 

ecoID = data[:,0]
eco = data[:,1]
nobserv = data[:,2]
maxBiom = data[:,3]
minBiom = data[:,4]
maxBA = data[:,5]
minBA = data[:,6]

BioVarsAll = numpy.arange(1, 20)

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


NEcoregions = numpy.size(eco)

vec = minBiom 
a = numpy.unique(vec)

# weights = numpy.array([])

for i in range(numpy.size(a)):
    ind = numpy.where(vec == a[i])
    print(eco[ind])
#    weights = numpy.append(weights, numpy.sum(nobserv[ind]))
#    print(a[i], eco[ind], nobserv[ind], numpy.sum(nobserv[ind])) 

# totalweight = numpy.sum(weights)

counts = numpy.array(numpy.unique(vec, return_counts=True)).T

for i in range(numpy.size(a)):
    ind = numpy.where(vec == a[i])
    print(counts[i][1])
#    w = round(numpy.sum(nobserv[ind])*100/totalweight, 1) 
#    print(a[i], nobserv[ind], numpy.sum(nobserv[ind]), w, counts[i][1], eco[ind]) 
    
for i in range(numpy.size(a)):
    name = BioVarsAllNames[a[i]-1]
    print(name)


    