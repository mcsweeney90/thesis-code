#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Similarity of empirical distributions for STG set.
"""

import pandas as pd
import numpy as np

runs = 100000
dists = ["normal", "gamma", "uniform"]
upper_dists = {"normal" : "NORMAL", "gamma" : "GAMMA", "uniform" : "UNIFORM"}
df = pd.read_csv('emp_stg.csv')

# =============================================================================
# Summaries.
# =============================================================================
   
with open("summary.txt", "w") as dest:
    print("SIMILARITY OF EMPIRICAL LONGEST PATH DISTRIBUTIONS WITH DIFFERENT WEIGHT DISTRIBUTIONS.", file=dest) 
    print("EMPIRICAL DISTRIBUTIONS GENERATED THROUGH {} REALIZATIONS OF GRAPH WEIGHTS.".format(runs), file=dest)          
                        
    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("% RELATIVE ERROR IN MEAN (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    ng = 100*abs(df["NORMAL MU"] - df["GAMMA MU"]) / df["GAMMA MU"]
    print("NORMAL-GAMMA: ({}, {})".format(ng.mean(), ng.max()), file=dest)
    nu = 100*abs(df["NORMAL MU"] - df["UNIFORM MU"]) / df["UNIFORM MU"]
    print("NORMAL-UNIFORM: ({}, {})".format(nu.mean(), nu.max()), file=dest)
    ug = 100*abs(df["UNIFORM MU"] - df["GAMMA MU"]) / df["GAMMA MU"]
    print("UNIFORM-GAMMA: ({}, {})".format(ug.mean(), ug.max()), file=dest)

    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("% RELATIVE ERROR IN VARIANCE (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    ng = 100*abs(df["NORMAL VAR"] - df["GAMMA VAR"]) / df["GAMMA VAR"]
    print("NORMAL-GAMMA: ({}, {})".format(ng.mean(), ng.max()), file=dest)
    nu = 100*abs(df["NORMAL VAR"] - df["UNIFORM VAR"]) / df["UNIFORM VAR"]
    print("NORMAL-UNIFORM: ({}, {})".format(nu.mean(), nu.max()), file=dest)
    ug = 100*abs(df["UNIFORM VAR"] - df["GAMMA VAR"]) / df["GAMMA VAR"]
    print("UNIFORM-GAMMA: ({}, {})".format(ug.mean(), ug.max()), file=dest)
    
    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("% RELATIVE ERROR IN STANDARD DEVIATION (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    ng = 100*abs(np.sqrt(df["NORMAL VAR"]) - np.sqrt(df["GAMMA VAR"])) / np.sqrt(df["GAMMA VAR"])
    print("NORMAL-GAMMA: ({}, {})".format(ng.mean(), ng.max()), file=dest)
    nu = 100*abs(np.sqrt(df["NORMAL VAR"]) - np.sqrt(df["UNIFORM VAR"])) / np.sqrt(df["UNIFORM VAR"])
    print("NORMAL-UNIFORM: ({}, {})".format(nu.mean(), nu.max()), file=dest)
    ug = 100*abs(np.sqrt(df["UNIFORM VAR"]) - np.sqrt(df["GAMMA VAR"])) / np.sqrt(df["GAMMA VAR"])
    print("UNIFORM-GAMMA: ({}, {})".format(ug.mean(), ug.max()), file=dest)
    
    print("\n\n\n------------------------------------------------------------------", file=dest)
    print("KS STATISTICS (AVG, MAX)", file=dest)
    print("------------------------------------------------------------------", file=dest)    
    print("NORMAL-GAMMA: ({}, {})".format(df["N-G KS"].mean(), df["N-G KS"].max()), file=dest)
    print("NORMAL-UNIFORM: ({}, {})".format(df["N-U KS"].mean(), df["N-U KS"].max()), file=dest)
    print("UNIFORM-GAMMA: ({}, {})".format(df["G-U KS"].mean(), df["G-U KS"].max()), file=dest)