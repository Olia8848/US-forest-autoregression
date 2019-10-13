

import numpy
import math
import scipy
import pandas   
import matplotlib.pyplot as plt
import pylab 


from numpy import random
from numpy import mean
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from scipy import stats


def inverseGamma(alpha, beta):
    return (1/numpy.random.gamma(alpha, 1/beta))


def Histograms(meansB, varsB, t, path2, Ecoregions, eid):
    msB = numpy.array([])
    vsB = numpy.array([])

    for j in range(NSIMS):
        msB = numpy.append(msB, meansB[t][j])      
        vsB = numpy.append(vsB, varsB[t][j])
    time = Years[t]
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    n, bins, patches = ax.hist(msB, bins=50, normed=False)
    ax.legend()
    
    n, bins, patches = ax2.hist(
    msB, cumulative=1, histtype='step', bins=100, color='tab:orange', label="CDF")
    ax2.legend(loc=2)
    
#    plt.hist(msB, bins = 50) 
#    plt.title("simulated means \n of ln(biomass) in " + str(time))
    plt.title("simulated Bayesian means \n of ln(BA) in " + str(time))
    plt.savefig(path2 + str(Ecoregions[eid]) + '_' + 'Means_Bayes_Hist_first_Year.png')
    plt.show()
    
    stats.probplot(msB, dist="norm", plot=pylab)
    pylab.title('simulated Bayesian means quantiles')
    pylab.show()
    

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    n, bins, patches = ax.hist(vsB, bins=50, normed=False)
    ax.legend()
    
    n, bins, patches = ax2.hist(
    vsB, cumulative=1, histtype='step', bins=100, color='tab:orange', label="CDF")
    ax2.legend(loc=2)
    
#    plt.hist(msB, bins = 50) 
#    plt.title("simulated means \n of ln(biomass) in " + str(time))
    plt.title("simulated Bayesian variances \n of ln(BA) in " + str(time))
    plt.savefig(path2 + str(Ecoregions[eid]) + '_' + 'Vars_Bayes_Hist_first_Year.png')
    plt.show()

    stats.probplot(vsB, dist="norm", plot=pylab)
    pylab.title('simulated Bayesian variances quantiles')
    pylab.show()
    


path = 'C:/Users/olga/Desktop/US_forest/'
path2 = 'C:/Users/olga/Desktop/US_forest/histograms_b_area_cdf_quantiles/'
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


Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions) # 36 ecoregions

###########################################################################
#########   Restricted by ecoregion: ######################################
###########################################################################

# for eid in range(NEcoregions):
#    print(eid)
eid = 0
ecoreg = numpy.array([str(data0[k,4]) for k in range(NOBSERV0)])  # Eco region
 
Ecoregions  = numpy.unique(ecoreg)
NEcoregions = numpy.size(Ecoregions)

df2 = df1[ecoreg == Ecoregions[eid]]

data = df2.values  

NOBSERV = numpy.size(data, 0) # in year Years[t]
print(NOBSERV)  # observations in your ecoregion


patch = data[:,0]   # Plot ID 
year = numpy.array([int(data[k,1]) for k in range(NOBSERV)]) # Year 
biomass = data[:,2] # Biomass
barea = data[:,3]   # Basal Area
ecoreg = numpy.array([str(data[k,4]) for k in range(NOBSERV)])  # Eco region
state = numpy.array([str(data[k,5]) for k in range(NOBSERV)])   # US State
# value = biomass # this is what we consider now: biomass 
value = biomass # this is what we consider now: biomass 

Years  = numpy.unique(year)
NYears = numpy.size(Years)

Patches  = numpy.unique(patch)
NPatches = numpy.size(Patches)



LogValuePatchYear = [[0] * NPatches for i in range(NYears)] 
# [[p1,..., pm]  [p1,..., pm] .... [p1,..., pm]]
#     year_1       year_2             year_n


#   LogValuePatchYear[i][p] = log(biomass at plot p in year i): 
for k in range(NOBSERV):
    indY = numpy.where(Years == year[k])
    i = indY[0][0]     # return i such that year[k] == Years[i]
    indP = numpy.where(Patches == patch[k])
    j = indP[0][0]  # return j such that patch[k] == Patches[j]
    LogValuePatchYear[i][j] = math.log(value[k])
#   i - year, j - patch

#  LogValuePatchYear[year][patch]
 
############################################################################
############################################################################
############################################################################  

NSIMS = 1000 # Number of simulations of Bayesian estimates

# means simulated by Bayes:
meansB =  [[0] * NSIMS for i in range(NYears)] 
# [[m_1,..., m_1000]  .... [m_1,..., m_1000]]
#     year_1                  year_38
#  meansB[year][simulation]


# vars simulated by Bayes:
varsB = [[0] * NSIMS for i in range(NYears)]
#  varsB[year][simulation]


for t in range(NYears):    
    LogValuesYear = numpy.array([])  # biomasses vector x_{1}(t), ... , x_{#p}(t)
    for p in range(NPatches):    
        LogValuesYear = numpy.append(LogValuesYear, LogValuePatchYear[t][p])
    LogValuesYear = LogValuesYear[numpy.nonzero(LogValuesYear)] #  biomass data doesn't  contain 0 records, so we can remove zeroes:
#    print('year = ', t)
    empMean = numpy.mean(LogValuesYear) # mean of biomasses in year t
    empVar = numpy.var(LogValuesYear) # variance of biomasses in year t
#    print('mean = ', round(empMean, 2))
#    print('var = ', round(empVar, 2))
    n = len(LogValuesYear)
    if n == 1:
        for j in range(NSIMS):
           varsB[t][j] = empVar 
           meansB[t][j] = empMean

    if n != 1:
        for j in range(NSIMS):
            varsB[t][j] = inverseGamma((n-1)/2, (n*empVar)/2) 
            meansB[t][j] = random.normal(empMean, varsB[t][j]/n)    
 

Histograms(meansB, varsB, 0, path2, Ecoregions, eid)
        
