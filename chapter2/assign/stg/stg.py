#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Critical assignment for STG graphs.
"""

import dill, os
import pandas as pd
from timeit import default_timer as timer # Haven't bothered to time since very similar to EFT.

import sys
sys.path.append('../../') 
from src import DAG, priority_scheduling

size = 100
dag_path = '../../../graphs/STG/{}'.format(size)

ccrs = [10.0]#[0.01, 0.1, 1.0, 10.0]
r = 32
ngpus = [1, 4]
runs = 1

# =============================================================================
# Get data.
# =============================================================================

data = [] 
for dname in os.listdir(dag_path):     
    print(dname)
    
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
                
                graph_data = {"r" : r, "s" : s, "CCR" : ccr, "NAME" : dname[:-5], "RUN" : run} 
                
                # Compute makespan lower bound.
                lb = G.makespan_lower_bound(r + s)
                graph_data["MLB"] = lb 
                
                # Minimal serial time.
                mst = G.minimal_serial_time()
                graph_data["MST"] = mst 
                
                # Compute standard upward ranks.
                prios = G.get_upward_ranks(r, s, avg_type="M") 
                
                # EFT.
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
    
df = pd.DataFrame(data)  
df.to_csv('stg2.csv', encoding='utf-8', index=False)