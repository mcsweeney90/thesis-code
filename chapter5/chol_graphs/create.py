#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Cholesky graphs.
TODO: will be far too expensive.
"""

import dill, pathlib
import numpy as np
import sys
sys.path.append('../') 
from src import RV, TDAG

####################################################################################################

dag_top_path = '../../graphs/cholesky/topologies/'
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nbs = [128, 1024]

timings_path = 'timings/' # TODO.
with open('{}/timings.dill'.format(timings_path), 'rb') as file:
    timings = dill.load(file)
runs = len(timings[128]["G"]["c"]) 

####################################################################################################


for nt in ntiles:  
    print("nt = {}".format(nt))
    # Get the graph topology.
    with open('{}/{}.dill'.format(dag_top_path, nt), 'rb') as file:
        T = dill.load(file)  
        
    for nb in nbs:     
        
        G = TDAG(T) # TODO.
        
        for t in G.top_sort:
            task_type = t[0]
            cmu = np.mean(timings[nb][task_type]["c"])
            cvar = np.var(timings[nb][task_type]["c"])
            gmu = np.mean(timings[nb][task_type]["g"])
            gvar = np.var(timings[nb][task_type]["g"])
            G.graph.nodes[t]['weight'] = {"c" : RV(cmu, cvar), "g" : RV(gmu, gvar)}
        
                
        # Set the communication costs for the tile size. TODO.
        y = 0
        
        # Save the DAG.
        dag_save_path = 'nb{}/'.format(nb)
        pathlib.Path(dag_save_path).mkdir(parents=True, exist_ok=True)
        with open('{}/{}.dill'.format(dag_save_path, nt), 'wb') as handle:
            dill.dump(G, handle)
                

