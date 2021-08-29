#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lookahead for STG set.
"""

import dill, os
import pandas as pd
from timeit import default_timer as timer

import sys
sys.path.append('../../') 
from src import DAG, priority_scheduling

size = 1000
dag_path = '../../../graphs/STG/{}'.format(size)

ccrs = [0.01, 0.1, 1.0, 10.0]
r = 32
ngpus = [1, 4]
runs = 10

# =============================================================================
# Get data.
# =============================================================================

data = [] 
for dname in os.listdir(dag_path):     
    # print(dname)
    # Load the DAG topology.
    with open('{}/{}'.format(dag_path, dname), 'rb') as file:
        T = dill.load(file)
    # Convert to DAG object.
    G = DAG(T)
    
    for s in ngpus:
        for ccr in ccrs:
            for run in range(runs):     
                
                # Set the weights.    
                G.set_random_weights(r=r, s=s, ccr=ccr)    
                
                graph_data = {"n" : G.size, "r" : r, "s" : s, "CCR" : ccr, "NAME" : dname[:-5], "RUN" : run} 
                
                # Compute makespan lower bound.
                lb = G.makespan_lower_bound(r + s)
                graph_data["MLB"] = lb 
                
                # Minimal serial time.
                mst = G.minimal_serial_time()
                graph_data["MST"] = mst 
                
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
    
df = pd.DataFrame(data)  
df.to_csv('stg{}.csv'.format(size), encoding='utf-8', index=False)