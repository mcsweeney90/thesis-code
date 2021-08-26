#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of timing data.
"""

import pathlib, dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from math import sqrt
from itertools import product
from scipy.stats import skew, kurtosis, kstest, ks_2samp

####################################################################################################

# Set some parameters for plots.
# See here: http://www.futurile.net/2016/02/27/matplotlib-beautiful-plots-with-style/
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlepad'] = 0
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.markersize'] = 3
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams["figure.figsize"] = (9.6,4)
plt.ioff() # Don't show plots.
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####################################################################################################

df = pd.read_csv('stg.csv')

summary_path = "summaries/time/"
pathlib.Path(summary_path).mkdir(parents=True, exist_ok=True)
plot_path = "plots/time/"
pathlib.Path(plot_path).mkdir(parents=True, exist_ok=True)

dists = ["normal", "gamma", "uniform"]
covs = [0.01, 0.03, 0.1, 0.3]

####################################################################################################

def summarize(data, name):
    with open("{}/time_{}.txt".format(summary_path, name), "w") as dest:
        print("SUMMARY OF STG TIMING.", file=dest)  
        
        print("\n\n\n------------------------------------------------------------------", file=dest)
        print("ALL VS CPM", file=dest)
        print("------------------------------------------------------------------", file=dest)        
        for method in ["CPM", "K", "SCULLI", "CORLCA", "MC10", "SIM10", "DOM10", "MC100", "SIM100", "DOM100"]:
            print("\n{}".format(method), file=dest)
            T = data["{} TIME".format(method)] / data["CPM TIME"]
            print("AVERAGE: {}".format(T.mean()), file=dest)
            print("MAX: {}".format(T.max()), file=dest)    
        
        # RPM. 
        print("\n\n\n------------------------------------------------------------------", file=dest)
        print("RPM VS CORLCA", file=dest)
        print("------------------------------------------------------------------", file=dest)
        
        for method in [ "MC10", "SIM10", "DOM10", "MC100", "SIM100", "DOM100"]:      
            print("\n{}".format(method), file=dest)
            T = data["{} TIME".format(method)] / data["CORLCA TIME"]
            print("AVERAGE: {}".format(T.mean()), file=dest)
            print("MAX: {}".format(T.max()), file=dest)
        
        
        
# Entire data set 
summarize(df, name="all")
# # By cov.
# for cov in covs:
#     sdf = df.loc[(df['COV'] == cov)] 
#     summarize(sdf, name="cov{}".format(cov))