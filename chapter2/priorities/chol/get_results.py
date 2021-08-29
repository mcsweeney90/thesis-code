#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task prioritization for Cholesky graphs.
"""

import dill
import pandas as pd

import sys
sys.path.append('../../') 
from src import average, priority_scheduling

####################################################################################################

dag_path = '../chol_graphs/'
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nbs = [128, 1024]
r = 32
ngpus = [1, 4]
u_avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]
o_avgs = ["M", "MD", "B", "W", "HM", "GM", "R", "D", "NC", "SD"]

# =============================================================================
# Get the data.
# =============================================================================

data = []
for nb in nbs:
    print("\nTile size: {}".format(nb))
    for N in ntiles:  
        print("DAG: {} x {}".format(N, N))
        # Load the DAG.
        with open('{}/nb{}/{}.dill'.format(dag_path, nb, N), 'rb') as file:
            G = dill.load(file)  
            
        # Compute minimal serial time.
        mst = G.minimal_serial_time()
        
        for s in ngpus:
            
            graph_data = {"r" : r, "s" : s, "n" : G.size, "N" : N, "NB" : nb, "MST" : mst} 
                
            # Compute makespan lower bound.
            lb = G.makespan_lower_bound(r + s)
            graph_data["MLB"] = lb      
            
            # Compute makespan for random selection policy.
            rand_prios = {t : G.size - i for i, t in enumerate(G.top_sort)}
            mkspan = priority_scheduling(G, r, s, priorities=rand_prios, sel_policy="EFT")
            graph_data["RND"] = mkspan
                        
            # Compute makespan for each average (upward ranks).
            for avg in u_avgs:  
                U = G.get_upward_ranks(r, s, avg_type=avg)
                mkspan = priority_scheduling(G, r, s, priorities=U, sel_policy="EFT")
                graph_data["{}-U".format(avg)] = mkspan
                        
            # Compute the makespans for each average (optimistic costs).
            OCT = G.optimistic_cost_table() 
            for avg in o_avgs:  
                ranks = {t : average(OCT[t]["c"] + G.graph.nodes[t]['weight']["c"], OCT[t]["g"] + G.graph.nodes[t]['weight']["g"], 
                              r=r, s=s, avg_type=avg) for t in G.top_sort} 
                mkspan = priority_scheduling(G, r, s, priorities=ranks, sel_policy="EFT") 
                graph_data["{}-O".format(avg)] = mkspan             
        
            data.append(graph_data)
                    
# # Save the dataframe. Commented out by default.
# df = pd.DataFrame(data)  
# df.to_csv('results.csv', encoding='utf-8', index=False)
