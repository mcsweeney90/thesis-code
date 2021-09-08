#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison for Cholesky DAGs.
NOTE: to avoid overwriting the data 'results.csv' that was used in thesis, have changed the name of save destination to 'new_results.csv'. 
ALSO: 
    1. This is very slow, especially for the larger graphs, so would not recommend running again.
    2. Saves full results and a "streamlined" version with just q = 20 and same V value for all coefficient of variation parameters.
"""

import dill
import pandas as pd
import numpy as np
from itertools import product
from timeit import default_timer as timer

import sys
sys.path.append('../') 
from src import DAG, priority_scheduling, cpop

####################################################################################################

dag_path = '../../graphs/cholesky/topologies/'
ntiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
nprocessors = [4, 20]
ccrs = [0.01, 0.1, 1.0, 10.0]
Vs = [0.2, 1.0]
runs = 3
avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]

# =============================================================================
# Get the data.
# =============================================================================

data = []
for N in ntiles:  
    # Load the DAG topology.
    with open('{}/{}.dill'.format(dag_path, N), 'rb') as file:
        T = dill.load(file)
    G = DAG(T)
               
    # for q, ccr, vproc, vrel, vband in product(nprocessors, ccrs, proc_cov, rel_cov, band_cov): 
    for q, ccr, v1, v2, v3 in product(nprocessors, ccrs, Vs, Vs, Vs): 
        for r in range(runs):      
            
            graph_data = {"n" : G.size, "N" : N, "run" : r, "q" : q, "CCR" : ccr, "vproc" : v1, "vrel" : v2, "vband" : v3} 
            
            # Set the costs.    
            G.set_cholesky_weights(nprocessors=q, ccr=ccr, vproc=v1, vrel=v2, vband=v3)  
            
            # Compute actual CCR.
            act_ccr = G.ccr()
            graph_data["Act. CCR"] = act_ccr 
            
            # Compute minimal serial time.
            mst = G.minimal_serial_time()  
            graph_data["MST"] = mst
                
            # Compute makespan lower bound.
            lb = G.makespan_lower_bound()
            graph_data["MLB"] = lb      
            
            # Compute makespan for random selection policy.
            rand_prios = {t : G.size - i for i, t in enumerate(G.top_sort)}
            mkspan = priority_scheduling(G, priorities=rand_prios)
            graph_data["RND-H"] = mkspan
            
            # Equivalent for CPOP.
            start = timer()
            cpop_prios = G.get_upward_ranks(avg_type="M") 
            elapsed = timer() - start
            graph_data["HEFT PRIO TIME"] = elapsed
            
            # Get a random path.
            ct = G.top_sort[0]
            path = [ct]
            while True:
                children = list(G.graph.successors(ct))
                if not children:
                    break
                child = np.random.choice(children)
                path.append(child)
                ct = child
            mkspan = cpop(G, priorities=cpop_prios, critical_path=path) 
            graph_data["RND-C"] = mkspan
                        
            # =============================================================================
            # Averages.   
            # =============================================================================
            for avg in avgs: 
                # HEFT.
                U = G.get_upward_ranks(avg_type=avg)
                heft_mkspan = priority_scheduling(G, priorities=U)
                graph_data["{}-H".format(avg)] = heft_mkspan 
                
                # CPOP.
                D = G.get_downward_ranks(avg_type=avg)
                ranks = {t : U[t] + D[t] for t in G.top_sort} 
                # Identify a single critical path (unless all_critical_tasks) - randomly if there are multiple...
                length = ranks[G.top_sort[0]] # Single entry/exit task.    
                ct = G.top_sort[0]
                path = [ct]
                while True:
                    children = list(G.graph.successors(ct))
                    if not children:
                        break
                    for child in children:
                        if abs(ranks[child] - length) < 1e-6:
                            path.append(child)
                            ct = child
                            break                 
                cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=path) 
                graph_data["{}-C".format(avg)] = cpop_mkspan      

            # =============================================================================
            # Critical path bounds.           
            # =============================================================================
            
            opt, opt_path = G.optimistic_critical_path(return_path=True)
            opt_ranks = {t : min(opt[t].values()) for t in G.top_sort}
            heft_mkspan = priority_scheduling(G, priorities=opt_ranks)
            graph_data["LB-H"] = heft_mkspan    
            cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=opt_path)        
            graph_data["LB-C"] = cpop_mkspan
            
            # Pessimistic critical paths.
            pes, pes_path = G.optimistic_critical_path(pessimistic=True, return_path=True)
            pes_ranks = {t : max(opt[t].values()) for t in G.top_sort}
            heft_mkspan = priority_scheduling(G, priorities=pes_ranks)
            graph_data["UB-H"] = heft_mkspan      
            cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=pes_path)        
            graph_data["UB-C"] = cpop_mkspan
                      
            # =============================================================================
            # Monte Carlo.           
            # =============================================================================
            for pmf in ["A", "H"]: 
                for r in [10, 1000]: # Do 10 new realizations rather than just sampling 10 in order to time etc.
                
                    L, path_counts, criticalities, timings = G.monte_carlo(realizations=r, pmf=pmf, times=True) 
                    
                    for k, v in timings.items():
                        graph_data["{}{}-{} TIME".format(pmf, r, k)] = v  
                    
                    # Number of critical paths.
                    paths = len(path_counts)
                    graph_data["{}{} PATHS".format(pmf, r)] = paths   
                    
                    s = "10" if r == 10 else ""
                    
                    # Criticalities.
                    mkspan = priority_scheduling(G, priorities=criticalities)
                    # graph_data["{}{}-C".format(pmf, r)] = mkspan 
                    graph_data["CR-{}{}".format(pmf, s)] = mkspan 
                    
                    # CPOP (most frequently critical path). 
                    mcp = max(path_counts, key=path_counts.get)
                    mkspan = cpop(G, priorities=cpop_prios, critical_path=mcp) 
                    # graph_data["{}{}-P".format(pmf, r)] = mkspan
                    graph_data["MCP-{}{}".format(pmf, s)] = mkspan
                    
                    # Mean.
                    means = {t: np.mean(L[t]) for t in G.top_sort}
                    mkspan = priority_scheduling(G, priorities=means)
                    graph_data["EV-{}{}".format(pmf, s)] = mkspan   
                    
                    # UCB.
                    ucbs = {t: np.mean(L[t]) + np.std(L[t]) for t in G.top_sort}
                    mkspan = priority_scheduling(G, priorities=ucbs)
                    graph_data["UCB-{}{}".format(pmf, s)] = mkspan    
                    
            # Save the data.
            data.append(graph_data)
                    
# # Save the dataframe.
df = pd.DataFrame(data)  
df.to_csv('new_results_full.csv', encoding='utf-8', index=False)
# Save streamlined version. 
rdf = df.loc[(df['q'] == 20) & ((df['vproc'] == 0.2) & (df['vrel'] == 0.2) & (df['vband'] == 0.2) | (df['vproc'] == 1.0) & (df['vrel'] == 1.0) & (df['vband'] == 1.0))]
rdf.to_csv('new_results.csv', encoding='utf-8', index=False)

