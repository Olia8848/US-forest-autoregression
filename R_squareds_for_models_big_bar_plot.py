# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:12:47 2020

@author: olga
"""

import pandas
import matplotlib.pyplot as plt
import numpy as np


#   Max Temperature of Warmest Month (BIO5)   	2.98
#   Precipitation of Coldest Quarter (BIO19)    7.67
#   Min Temperature of Coldest Month (BIO6)     7.83
#   Temperature Seasonality (BIO4)              7.84

#  Precipitation Seasonality (BIO15)	        38.81
#  Mean Temperature of Driest Quarter (BIO9)   	64.42
#  Mean Diurnal Range (BIO2)  	            	68.44
#  Precipitation of Driest Quarter (BIO17)   	71.19


df = pandas.DataFrame(dict(graph=['\n \n trivial \n regressions', '\n \n regressions \n with Bayesian \n simulations'],
                           n1=[2.98, 38.81],  #  or, bl
                           n2=[7.67, 64.42],  #  bl, or
                           n3=[7.83, 68.44],  #  or, viol
                           n4=[7.84, 71.19],)) # viol, bl
ind = np.arange(len(df))
width = 0.23

fig, ax = plt.subplots()
ax.barh(ind, df.n1, width, color=["orange", "blue"], label='N1')
ax.barh(ind + width, df.n2, width, color=["blue", "orange"], label='N2')
ax.barh(ind + width + width, df.n3, width, color=["orange", "purple"], label='N3')
ax.barh(ind + width + width + width, df.n4, width, color=["purple", "blue"], label='N4')


ax.set(yticks=ind + width + width + width, yticklabels=df.graph, ylim=[4*width - 1, len(df)])
# ax.legend()
plt.show()
