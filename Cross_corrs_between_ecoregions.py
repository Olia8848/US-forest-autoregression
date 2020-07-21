# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:29:23 2019

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


from itertools import combinations 

def combine(arr, s): 
    return list(combinations(arr, s)) 

def inverseGamma(alpha, beta):
    return (1/numpy.random.gamma(alpha, 1/beta))

def symbayesestimates(df1, eco, NSIMS, indicator):
    data0 = df1.values  
    NOBSERV0 = numpy.size(data0, 0) 
    patch = data0[:,0]   # Plot ID 
    year = numpy.array([int(data0[k,1]) for k in range(NOBSERV0)]) # Year 
    biomass = data0[:,2] # Biomass
    barea = data0[:,3]   # Basal Area
    ecoreg = numpy.array([str(data0[k,4]) for k in range(NOBSERV0)])  # Eco region

    Ecoregions  = numpy.unique(ecoreg)

    df2 = df1[ecoreg == Ecoregions[eco]]
    data = df2.values  

    NOBSERV = numpy.size(data, 0) # in year Years[t]

    patch = data[:,0]   # Plot ID 
    year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) # Year 
    biomass = data[:,2] # Biomass
    barea = data[:,3]   # Basal Area
    ecoreg = numpy.array([str(data[k,4]) for k in range(NOBSERV)])  # Eco region

    if indicator == 1:
       value = biomass # this is what we consider now: biomass
    if indicator == 0:
       value = barea 

    Years  = numpy.unique(year)
    NYears = numpy.size(Years)

    Patches  = numpy.unique(patch)
    NPatches = numpy.size(Patches)

    LogValuePatchYear = [[0] * NPatches for i in range(NYears)] 

    for k in range(NOBSERV):
        indY = numpy.where(Years == year[k])
        i = indY[0][0]     # return i such that year[k] == Years[i]
        indP = numpy.where(Patches == patch[k])
        j = indP[0][0]  # return j such that patch[k] == Patches[j]
        LogValuePatchYear[i][j] = math.log(value[k])


    meansB =  [[0] * NSIMS for i in range(NYears)] 
    varsB = [[0] * NSIMS for i in range(NYears)]

    for t in range(NYears):    
        LogValuesYear = numpy.array([])  # biomasses vector x_{1}(t), ... , x_{#p}(t)
        for p in range(NPatches):    
            LogValuesYear = numpy.append(LogValuesYear, LogValuePatchYear[t][p])
        LogValuesYear = LogValuesYear[numpy.nonzero(LogValuesYear)] #  biomass data doesn't  contain 0 records, so we can remove zeroes:

        empMean = numpy.mean(LogValuesYear) # mean of biomasses in year t
        empVar = numpy.var(LogValuesYear) # variance of biomasses in year t

        n = len(LogValuesYear) # number of patches observed in year t
        if n == 1:
            continue
#        print(Years[t], ' & number of patches observed in this year: ', n, ' & empir. mean: ', round(empMean, 2), ' & empir. variance:', round(empVar, 2))
        for j in range(NSIMS):
            varsB[t][j] = inverseGamma((n-1)/2, (n*empVar)/2) 
            meansB[t][j] = random.normal(empMean, varsB[t][j]/n)
# g^hat - estimate:
    summ = 0
    for i in range(NSIMS):
        summ = summ + meansB[NYears-1][i]-meansB[0][i]  

    ghat = summ/(NSIMS*NYears)


    # sigma^hat - estimate:
    summ = 0
    for i in range(NSIMS):
        for t in range(NYears):
            summ = summ + (meansB[t][i]-meansB[t-1][i] - ghat)**2
    
    sigmaHat = numpy.sqrt(summ/(NSIMS*NYears))    

    
    return meansB, varsB, ghat, sigmaHat, Ecoregions, Years

#####################################################################    
    
path = 'C:/Users/olga/Desktop/US_forest/'
# path = '/Users/Olga Rumyantseva/Desktop/Python biomass/'
df = pandas.read_csv(path + 'biodataUS_sorted.csv')


df.columns

df1 = df.iloc[:, [2, 3, 18, 19, 24, 25, 22, 23]]
df1.columns   

# document = Document()
# document.add_heading('Cross correlations between two ecoregions:')

NSIMS = 1000 # Number of simulations of Bayesian estimates 

indicator = 0 # run code for biomass (1) or barea (0)
 
Ecoregions = symbayesestimates(df1, 0, NSIMS, indicator)[4]
# symbayesestimates(df1, eco, NSIMS, indicator)
# returns meansB, varsB, ghat, sigmaHat, Ecoregions, Years

# Ghats = numpy.array([])
# for i in range(36):
#     ghat = symbayesestimates(df1, i, NSIMS, indicator)[2]
#     Ghats = numpy.append(Ghats, ghat)


# sigma-cross-corr - estimate:

EcoCombinations = combine(range(36), 2)
N = numpy.size(EcoCombinations)

A = [[0] * 36 for i in range(36)] # cross-corrs of biomass


for k in range(630): # 630

    comb = EcoCombinations[k]
    eco1 = comb[0]
    eco2 = comb[1]

    res1 = symbayesestimates(df1, eco1, NSIMS, indicator)
    res2 = symbayesestimates(df1, eco2, NSIMS, indicator)
  # returns meansB, varsB, ghat, sigmaHat, Ecoregions, Years

    meansB1 = res1[0]
    meansB2 = res2[0]

    ghat1 = res1[2]
    ghat2 = res2[2]
    
    sigmaHat1 = res1[3]
    sigmaHat2 = res2[3]

    Years1 = res1[5]
    Years2 = res2[5]

    CommonYears = numpy.intersect1d(Years1, Years2)
    T = numpy.size(CommonYears)
    if T == 0:
       continue

    summ = 0
    for i in range(NSIMS):
        for t in range(T):
            i1 = numpy.where(Years1 == CommonYears[t])
            t1 = i1[0][0]
            i2 = numpy.where(Years2 == CommonYears[t])
            t2 = i2[0][0]
            print(t, t1, t2)
            d1 = (meansB1[t1][i]-meansB1[t1-1][i] - ghat1)
            d2 = (meansB2[t2][i]-meansB2[t2-1][i] - ghat2)
            summ = summ + d1*d2

    CrossCorr = summ/(NSIMS*T*sigmaHat1*sigmaHat2) 

    A[eco1][eco2] = CrossCorr
    A[eco2][eco1] = CrossCorr
#    document.add_paragraph('Iteration ' + str(k) + ',  Ecoreg 1: ' + ' ' + str(Ecoregions[eco1]) + ',  Ecoreg 2: ' + ' ' + str(Ecoregions[eco2]) + '\n' + 'Cross Corr.: ' + str(CrossCorr) + '\n' + 'Common Years: ' + str(CommonYears))
#    document.add_paragraph('  ')
    print('Iteration  ', k, ',   Ecoreg 1:  ', Ecoregions[eco1], ',  Ecoreg 2: ', Ecoregions[eco2], ', Cross Corr.: ', CrossCorr, '\n Common Years: ', CommonYears) 
    
# document.save(path + 'Cross_correlations_between_two_ecoregions.docx')

pandas.DataFrame(A).to_csv("C:/Users/olga/Desktop/US_forest/tables/CrossCorrelations_BArea.csv")


#############################################################################
#############################################################################


indicator = 1 # run code for biomass (1) or barea (0)



B = [[0] * 36 for i in range(36)] # cross-corrs of biomass


for k in range(630):
    comb = EcoCombinations[k]
    eco1 = comb[0]
    eco2 = comb[1]

    res1 = symbayesestimates(df1, eco1, NSIMS, indicator)
    res2 = symbayesestimates(df1, eco2, NSIMS, indicator)
 # returns meansB, varsB, ghat, sigmaHat, Ecoregions, Years
 
    meansB1 = res1[0]
    meansB2 = res2[0]

    ghat1 = res1[2]
    ghat2 = res2[2]
    
    sigmaHat1 = res1[3]
    sigmaHat2 = res2[3]

    Years1 = res1[5]
    Years2 = res2[5]

    CommonYears = numpy.intersect1d(Years1, Years2)
    T = numpy.size(CommonYears)
    if T == 0:
       continue

    summ = 0
    for i in range(NSIMS):
        for t in range(T):
            i1 = numpy.where(Years1 == CommonYears[t])
            t1 = i1[0][0]
            i2 = numpy.where(Years2 == CommonYears[t])
            t2 = i2[0][0]
            d1 = (meansB1[t1][i]-meansB1[t1-1][i] - ghat1)
            d2 = (meansB2[t2][i]-meansB2[t2-1][i] - ghat2)
            summ = summ + d1*d2

    CrossCorr = summ/(NSIMS*T*sigmaHat1*sigmaHat2) 
    B[eco1][eco2] = CrossCorr
    B[eco2][eco1] = CrossCorr
#    document.add_paragraph('Iteration ' + str(k) + ',  Ecoreg 1: ' + ' ' + str(Ecoregions[eco1]) + ',  Ecoreg 2: ' + ' ' + str(Ecoregions[eco2]) + '\n' + 'Cross Corr.: ' + str(CrossCorr) + '\n' + 'Common Years: ' + str(CommonYears))
#    document.add_paragraph('  ')
    print('Iteration  ', k, ',   Ecoreg 1:  ', Ecoregions[eco1], ',  Ecoreg 2: ', Ecoregions[eco2], ', Cross Corr.: ', CrossCorr, '\n Common Years: ', CommonYears) 
    
# document.save(path + 'Cross_correlations_between_two_ecoregions.docx')

pandas.DataFrame(B).to_csv("C:/Users/olga/Desktop/US_forest/tables/CrossCorrelations_Biomass.csv")
