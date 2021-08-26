#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timing MC method. 
"""

import dill
import pandas as pd
from itertools import product
from timeit import default_timer as timer

# TODO: the following are apparently needed to run the script on Matt's machine - why?
import sys
sys.path.append("../")

runs = 10
data = []
ntasks = list(range(5, 51, 5))
for nt, nb, s in product(ntasks, [128, 1024], [1, 4]): 
    chol_load_path = '../chol_graphs/nb{}s{}'.format(nb, s)
    with open('{}/{}.dill'.format(chol_load_path, nt), 'rb') as file:
        G = dill.load(file)
    graph_data = {"n" : G.size, "nt" : nt, "nb" : nb, "s" : s}
    
    start = timer()
    for _ in range(runs):
        cpm = G.CPM()
    elapsed = timer() - start
    graph_data["CPM"] = elapsed/runs 
        
    # Sculli.
    start = timer()
    for _ in range(runs):
        SL = G.sculli()
    elapsed = timer() - start
    graph_data["SCULLI"] = elapsed/runs 
    
    # CorLCA.
    start = timer()
    for _ in range(runs):
        CL = G.corLCA()
    elapsed = timer() - start
    graph_data["CORLCA"] = elapsed/runs 
    
    # Kamburowski.
    start = timer()
    for _ in range(runs):
        lm, um, ls, us = G.kamburowski()
    elapsed = timer() - start
    graph_data["K"] = elapsed/runs   
    
    # MC10.
    start = timer()
    for _ in range(runs):
        E = G.monte_carlo(samples=10, dist="U")
    elapsed = timer() - start
    graph_data["MC10"] = elapsed/runs     
    
    # MC100.
    start = timer()
    for _ in range(runs):
        E = G.monte_carlo(samples=100, dist="U")
    elapsed = timer() - start
    graph_data["MC100"] = elapsed/runs 
    
    # Save the data.
    data.append(graph_data)
        
# Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('timing.csv', encoding='utf-8', index=False)     

