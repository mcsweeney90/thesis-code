#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical assignment for Cholesky graphs.
"""

import dill
import pandas as pd
from timeit import default_timer as timer

import sys
sys.path.append('../../') 
from src import priority_scheduling

####################################################################################################

dag_path = '../chol_graphs/'
ntiles = [5, 10, 15, 20]#, 25, 30, 35, 40, 45, 50]
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
            
            # Compute upward ranks (to be used as task priorities in all cases).
            prios = G.get_upward_ranks(r, s, avg_type="M") 
            eft = priority_scheduling(G, r, s, priorities=prios, sel_policy="EFT")
            graph_data["EFT"] = eft
            
            for avg in ["M", "HM", "O"]: 
                if avg == "O":
                    critical_path = G.get_critical_path(cp_type="OPT")
                else:                    
                    critical_path = G.get_critical_path(cp_type="AVG", avg_type=avg, r=r, s=s)               
                # Assign as in CPOP.
                total_cpu_cost = sum(G.graph.nodes[ct]['weight']["c"] for ct in critical_path)
                total_gpu_cost = sum(G.graph.nodes[ct]['weight']["g"] for ct in critical_path)
                if total_cpu_cost < total_gpu_cost:
                    alpha = {ct : "c" for ct in critical_path}
                else:
                    alpha = {ct : r for ct in critical_path} # Entry task on CP so any choice will do.             
                indy = priority_scheduling(G, r, s, priorities=prios, sel_policy="AMT", assignment=alpha)
                graph_data["{}-C".format(avg)] = indy                                     
                
                # Compute best assignment for the critical path.                
                OCT, parent_assignment = {}, {}
                for i, ct in enumerate(critical_path):
                    OCT[ct] = {"c" : G.graph.nodes[ct]['weight']["c"], "g" : G.graph.nodes[ct]['weight']["g"]}
                    parent_assignment[ct] = {}
                    if i == 0:
                        continue
                    parent = critical_path[i - 1]
                    # CPU.
                    if OCT[parent]["c"] < OCT[parent]["g"] + G.graph[parent][ct]['weight']:
                        OCT[ct]["c"] += OCT[parent]["c"]
                        parent_assignment[ct]["c"] = "c"
                    else:
                        OCT[ct]["c"] += (OCT[parent]["g"] + G.graph[parent][ct]['weight'])
                        parent_assignment[ct]["c"] = "g"                    
                    # GPU.
                    if OCT[parent]["c"] + G.graph[parent][ct]['weight'] < OCT[parent]["g"] + ((s - 1)/s) * G.graph[parent][ct]['weight']:
                        OCT[ct]["g"] += (OCT[parent]["c"] + G.graph[parent][ct]['weight'])
                        parent_assignment[ct]["g"]  = "c"
                    else:
                        OCT[ct]["g"] += (OCT[parent]["g"] + ((s - 1)/s) * G.graph[parent][ct]['weight'])
                        parent_assignment[ct]["g"]  = "g"
                beta = {}                
                backward = list(reversed(critical_path))
                for i, ct in enumerate(backward):
                    if i == 0:
                        proc_type = "c" if OCT[ct]["c"] < OCT[ct]["g"] else "g"
                    else:
                        parent_assignment[backward[i - 1]][proc_type]
                    beta[ct] = proc_type                        
                diff = priority_scheduling(G, r, s, priorities=prios, sel_policy="AMT", assignment=beta)
                graph_data["{}-A".format(avg)] = diff 
                
            data.append(graph_data)
                    
#Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('chol.csv', encoding='utf-8', index=False)