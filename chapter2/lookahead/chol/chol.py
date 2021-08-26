#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lookahead for Cholesky graphs.
"""

import dill
import pandas as pd
from timeit import default_timer as timer

import sys
sys.path.append('../../') 
from src import priority_scheduling

####################################################################################################

dag_path = '../chol_graphs/'
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nbs = [128, 1024]
r = 32
ngpus = [1, 4]

# =============================================================================
# Get the data.
# =============================================================================

data = []
for nb in nbs:
    print("\nTile size: {}".format(nb))
    for nt in ntiles:  
        print("DAG: {}".format(nt))
        # Load the DAG.
        with open('{}/nb{}/{}.dill'.format(dag_path, nb, nt), 'rb') as file:
            G = dill.load(file)  
            
        # Compute minimal serial time.
        mst = G.minimal_serial_time()
        
        for s in ngpus:
            
            graph_data = {"r" : r, "s" : s, "n" : G.size, "nt" : nt, "NB" : nb, "MST" : mst}
                
            # Compute makespan lower bound.
            lb = G.makespan_lower_bound(r + s)
            graph_data["MLB"] = lb              
            
            # Compute standard upward ranks.
            U = G.get_upward_ranks(r, s, avg_type="M") 
            
            # EFT.
            start = timer()
            eft = priority_scheduling(G, r, s, priorities=U, sel_policy="EFT")
            eft_time = timer() - start
            graph_data["EFT"] = eft  
            
            # NC.
            start = timer()
            nc = priority_scheduling(G, r, s, priorities=U, sel_policy="NC")
            nc_time = timer() - start
            graph_data["NC"] = nc 
            graph_data["NC TIME"] = nc_time/eft_time
            
            # Optimistic lookahead, original version.
            start = timer()
            OCT1 = G.orig_optimistic_cost_table(r, s)
            ol1 = priority_scheduling(G, r, s, priorities=U, sel_policy="PEFT", lookahead_table=OCT1)
            ol1_time = timer() - start
            graph_data["OL-I"] = ol1
            graph_data["OL-I TIME"] = ol1_time/eft_time
            
            # Optimistic lookahead, slightly modified version.
            start = timer()
            OCT2 = G.optimistic_cost_table()
            ol2 = priority_scheduling(G, r, s, priorities=U, sel_policy="PEFT", lookahead_table=OCT2)
            ol2_time = timer() - start
            graph_data["OL-II"] = ol2 
            graph_data["OL-II TIME"] = ol2_time/eft_time
            
            # # Binary lookahead.
            start = timer()
            bl = priority_scheduling(G, r, s, priorities=U, sel_policy="BL")
            bl_time = timer() - start
            graph_data["BL"] = bl            
            graph_data["BL TIME"] = bl_time/eft_time
            
            # GCP - GPU communication penalty.
            start = timer()
            gcp = priority_scheduling(G, r, s, priorities=U, sel_policy="GCP")
            gcp_time = timer() - start
            graph_data["GCP"] = gcp 
            graph_data["GCP TIME"] = gcp_time/eft_time
            
            # HAL - harmonic average lookahead.
            start = timer()
            hal = priority_scheduling(G, r, s, priorities=U, sel_policy="HAL")
            hal_time = timer() - start
            graph_data["HAL"] = hal 
            graph_data["HAL TIME"] = hal_time/eft_time           
        
            data.append(graph_data)
                    
#Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('chol.csv', encoding='utf-8', index=False)

