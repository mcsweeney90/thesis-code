#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of STG results wrt timings.
"""

import pathlib
import matplotlib.pyplot as plt
import pandas as pd

####################################################################################################

# Set some parameters for plots.
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
plt.ioff() # Don't show plots.

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