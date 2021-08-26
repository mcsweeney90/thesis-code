#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task prioritization for STG set.
"""

import dill, os
import pandas as pd

import sys
sys.path.append('../../') 
from src import DAG, average, priority_scheduling

size = 1000
dag_path = '../../../graphs/STG/{}'.format(size)

ccrs = [0.01, 0.1, 1.0, 10.0]
r = 32
ngpus = [1, 4]
runs = 10
u_avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]
o_avgs = ["M", "MD", "B", "W", "HM", "GM", "R", "D", "NC", "SD"]

# =============================================================================
# Get data.
# =============================================================================

# data = [] 
# for dname in os.listdir(dag_path):     
    
#     # Load the DAG topology.
#     with open('{}/{}'.format(dag_path, dname), 'rb') as file:
#         T = dill.load(file)
#     # Convert to DAG object.
#     G = DAG(T)
    
#     for s in ngpus:
#         for ccr in ccrs:
#             for run in range(runs):     
                
#                 # Set the weights.    
#                 G.set_random_weights(r=r, s=s, ccr=ccr)    
                
#                 graph_data = {"n" : G.size, "r" : r, "s" : s, "CCR" : ccr, "NAME" : dname[:-5], "RUN" : run} 
                
#                 # Compute makespan lower bound.
#                 lb = G.makespan_lower_bound(r + s)
#                 graph_data["MLB"] = lb 
                
#                 # Minimal serial time.
#                 mst = G.minimal_serial_time()
#                 graph_data["MST"] = mst 
                
#                 # Compute makespan for random selection policy.
#                 rand_prios = {t : G.size - i for i, t in enumerate(G.top_sort)}
#                 mkspan = priority_scheduling(G, r, s, priorities=rand_prios, sel_policy="EFT")
#                 graph_data["RND"] = mkspan
                            
#                 # Compute makespan for each average (upward ranks).
#                 for avg in u_avgs:  
#                     U = G.get_upward_ranks(r, s, avg_type=avg)
#                     mkspan = priority_scheduling(G, r, s, priorities=U, sel_policy="EFT")
#                     graph_data["U-{}".format(avg)] = mkspan
                            
#                 # Compute the makespans for each average (optimistic costs).
#                 OCT = G.optimistic_cost_table() 
#                 for avg in o_avgs:  
#                     ranks = {t : average(OCT[t]["c"] + G.graph.nodes[t]['weight']["c"], OCT[t]["g"] + G.graph.nodes[t]['weight']["g"], 
#                                   r=r, s=s, avg_type=avg) for t in G.top_sort} 
#                     mkspan = priority_scheduling(G, r, s, priorities=ranks, sel_policy="EFT")   
#                     graph_data["O-{}".format(avg)] = mkspan
                
#                 data.append(graph_data)        
    
# df = pd.DataFrame(data)  
# df.to_csv('stg{}.csv'.format(size), encoding='utf-8', index=False)

# Rename averages (move U and O to the back).
df = pd.read_csv('stg{}.csv'.format(size))
cols = {}
for avg in u_avgs:
    cols["U-{}".format(avg)] = "{}-U".format(avg)
for avg in o_avgs:
    cols["O-{}".format(avg)] = "{}-O".format(avg)
df.rename(columns=cols, inplace=True)
df.to_csv('test.csv', encoding='utf-8', index=False)
                