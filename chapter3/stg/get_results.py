#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STG set. 
"""

import dill, os, csv
import numpy as np
from itertools import product
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from src import DAG, priority_scheduling, cpop

import pandas as pd

####################################################################################################

size = 100
dag_path = '../../graphs/STG/{}'.format(size)
nprocessors = [2, 4, 8]
ccrs = [0.01, 0.1, 1.0, 2.0]
Rs = [0.1, 0.5, 0.9]
Vs = [0.2, 1.0]
runs = 3
avgs = ["M", "MD", "B", "SB", "W", "SW", "HM", "SHM", "GM", "SGM", "R", "D", "NC", "SD"]

####################################################################################################

# CHANGE COLUMN TITLES.
# df = pd.read_csv('stg_highcomm.csv')
# df.rename(columns={"A1000-P" : "MCP-A", "H1000-P" : "MCP-H", "A1000-M" : "EV-A", "H1000-M" : "EV-H", "A1000-C" : "CR-A", "H1000-C" : "CR-H"}, inplace=True)

# sdf = df.loc[(df['mu_ccr'] == 2.0)] 
# sdf.to_csv('stg2.csv', encoding='utf-8', index=False)

# Combine the two files.
# df1 = pd.read_csv('stg.csv')
# df2 = pd.read_csv('stg2.csv')
# test = pd.concat([df1, df2], ignore_index=True)
# test.to_csv('test.csv', encoding='utf-8', index=False)

# =============================================================================
# Get the data.
# =============================================================================

# with open('stg.csv', 'w', encoding='UTF8') as f:
    
#     writer = csv.writer(f)
#     header = ["DAG", "q", "rtask", "rmach", "mu_ccr", "V", "run", "CCR", "MST", "MLB", "RND-H", "HEFT PRIO TIME", "RND-C"]
#     for avg in avgs:
#         header += ["{}-H".format(avg), "{}-C".format(avg)]
#     header += ["LB-H", "LB-C", "UB-H", "UB-C"]
#     for pmf in ["A", "H"]: 
#         for r in [10, 1000]: 
#             header += ["{}{}-REAL TIME".format(pmf, r), "{}{}-LONG TIME".format(pmf, r), "{}{}-CRT TIME".format(pmf, r),
#                        "{}{} PATHS".format(pmf, r), "{}{}-C".format(pmf, r), "{}{}-P".format(pmf, r), "{}{}-M".format(pmf, r),
#                        "{}{}-U".format(pmf, r)]  
#     writer.writerow(header)

#     for dname in os.listdir(dag_path):   
#         # print(dname)
        
#         # Load the DAG topology.
#         with open('{}/{}'.format(dag_path, dname), 'rb') as file:
#             T = dill.load(file)
#         # Convert to DAG object.
#         G = DAG(T)   
        
#         for q, rtask, rmach, muccr, v in product(nprocessors, Rs, Rs, ccrs, Vs):
#             for run in range(runs):   
#                 # graph_data = {"DAG" : dname[:-5], "q" : q, "rtask" : rtask, "rmach" : rmach, "mu_ccr" : muccr, "V" : v, "RUN" : run}
#                 data = [dname[:-5], q, rtask, rmach, muccr, v, run]
                
#                 # Set costs.
#                 params = (rtask, rmach, 1.0, v)
#                 G.set_random_weights(nprocessors=q, comp_method="CNB", comp_params=params, vband=v, muccr=muccr, vccr=v)            
                           
#                 # Compute actual CCR.
#                 act_ccr = G.ccr()
#                 # graph_data["CCR"] = act_ccr 
#                 data.append(act_ccr)
                
#                 # Compute minimal serial time.
#                 mst = G.minimal_serial_time()  
#                 # graph_data["MST"] = mst
#                 data.append(mst)
                    
#                 # Compute makespan lower bound.
#                 lb = G.makespan_lower_bound()
#                 # graph_data["MLB"] = lb    
#                 data.append(lb)
                
#                 # Compute makespan for random selection policy.
#                 rand_prios = {t : G.size - i for i, t in enumerate(G.top_sort)}
#                 mkspan = priority_scheduling(G, priorities=rand_prios)
#                 # graph_data["RND-H"] = mkspan
#                 data.append(mkspan)
                
#                 # Equivalent for CPOP.
#                 start = timer()
#                 cpop_prios = G.get_upward_ranks(avg_type="M") # TODO: best choice?
#                 elapsed = timer() - start
#                 # graph_data["HEFT PRIO TIME"] = elapsed
#                 data.append(elapsed)
                
#                 # Get a random path.
#                 ct = G.top_sort[0]
#                 path = [ct]
#                 while True:
#                     children = list(G.graph.successors(ct))
#                     if not children:
#                         break
#                     child = np.random.choice(children)
#                     path.append(child)
#                     ct = child
#                 mkspan = cpop(G, priorities=cpop_prios, critical_path=path) 
#                 # graph_data["RND-C"] = mkspan
#                 data.append(mkspan)
                            
#                 # =============================================================================
#                 # Averages.   
#                 # =============================================================================
#                 for avg in avgs: 
#                     # HEFT.
#                     U = G.get_upward_ranks(avg_type=avg)
#                     heft_mkspan = priority_scheduling(G, priorities=U)
#                     # graph_data["{}-H".format(avg)] = heft_mkspan 
#                     data.append(heft_mkspan)
                    
#                     # CPOP.
#                     D = G.get_downward_ranks(avg_type=avg)
#                     ranks = {t : U[t] + D[t] for t in G.top_sort} 
#                     # Identify a single critical path (unless all_critical_tasks) - randomly if there are multiple...
#                     length = ranks[G.top_sort[0]] # Single entry/exit task.    
#                     ct = G.top_sort[0]
#                     path = [ct]
#                     while True:
#                         children = list(G.graph.successors(ct))
#                         if not children:
#                             break
#                         for child in children:
#                             if abs(ranks[child] - length) < 1e-6:
#                                 path.append(child)
#                                 ct = child
#                                 break                 
#                     cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=path) 
#                     # graph_data["{}-C".format(avg)] = cpop_mkspan     
#                     data.append(cpop_mkspan)
    
#                 # =============================================================================
#                 # Critical path bounds.           
#                 # =============================================================================
                
#                 opt, opt_path = G.optimistic_critical_path(return_path=True)
#                 opt_ranks = {t : min(opt[t].values()) for t in G.top_sort}
#                 heft_mkspan = priority_scheduling(G, priorities=opt_ranks)
#                 # graph_data["LB-H"] = heft_mkspan
#                 data.append(heft_mkspan)                
#                 cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=opt_path)        
#                 # graph_data["LB-C"] = cpop_mkspan
#                 data.append(cpop_mkspan)
                
#                 # Pessimistic critical paths.
#                 pes, pes_path = G.optimistic_critical_path(pessimistic=True, return_path=True)
#                 pes_ranks = {t : max(opt[t].values()) for t in G.top_sort}
#                 heft_mkspan = priority_scheduling(G, priorities=pes_ranks)
#                 # graph_data["UB-H"] = heft_mkspan  
#                 data.append(heft_mkspan)
#                 cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=pes_path)        
#                 # graph_data["UB-C"] = cpop_mkspan
#                 data.append(cpop_mkspan)
                          
#                 # =============================================================================
#                 # Monte Carlo.           
#                 # =============================================================================
#                 for pmf in ["A", "H"]: 
#                     for r in [10, 1000]: # Do 10 new realizations rather than just sampling 10 in order to time etc.
                    
#                         L, path_counts, criticalities, timings = G.monte_carlo(realizations=r, pmf=pmf, times=True) 
                        
#                         for k, val in timings.items():
#                             # graph_data["{}{}-{} TIME".format(pmf, r, k)] = val 
#                             data.append(val)
                        
#                         # Number of critical paths.
#                         paths = len(path_counts)
#                         # graph_data["{}{} PATHS".format(pmf, r)] = paths 
#                         data.append(paths)
                        
#                         # Criticalities.
#                         mkspan = priority_scheduling(G, priorities=criticalities)
#                         # graph_data["{}{}-C".format(pmf, r)] = mkspan 
#                         data.append(mkspan)
                        
#                         # CPOP (most frequently critical path). 
#                         mcp = max(path_counts, key=path_counts.get)
#                         mkspan = cpop(G, priorities=cpop_prios, critical_path=mcp) 
#                         # graph_data["{}{}-P".format(pmf, r)] = mkspan
#                         data.append(mkspan)
                        
#                         # Mean.
#                         means = {t: np.mean(L[t]) for t in G.top_sort}
#                         mkspan = priority_scheduling(G, priorities=means)
#                         # graph_data["{}{}-M".format(pmf, r)] = mkspan  
#                         data.append(mkspan)
                        
#                         # UCB.
#                         ucbs = {t: np.mean(L[t]) + np.std(L[t]) for t in G.top_sort}
#                         mkspan = priority_scheduling(G, priorities=ucbs)
#                         # graph_data["{}{}-U".format(pmf, r)] = mkspan  
#                         data.append(mkspan)  
#                 # Write row to file.
#                 writer.writerow(data)


    
    

