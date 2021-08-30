#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get results for STG set. 
"""

import dill, os, csv
import numpy as np
from itertools import product
from timeit import default_timer as timer
import sys
sys.path.append('../') 
from src import DAG, priority_scheduling, cpop

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

# =============================================================================
# Generate results.
# =============================================================================

with open('new_results.csv', 'w', encoding='UTF8') as f: # Or whatever name for the file...
    
    writer = csv.writer(f)
    header = ["DAG", "q", "rtask", "rmach", "mu_ccr", "V", "run", "CCR", "MST", "MLB", "RND-H", "HEFT PRIO TIME", "RND-C"]
    for avg in avgs:
        header += ["{}-H".format(avg), "{}-C".format(avg)]
    header += ["LB-H", "LB-C", "UB-H", "UB-C"]
    for pmf in ["A", "H"]: 
        for r in [10, 1000]: 
            s = "10" if r == 10 else ""
            header += ["{}{}-REAL TIME".format(pmf, r), "{}{}-LONG TIME".format(pmf, r), "{}{}-CRT TIME".format(pmf, r),
                        "{}{} PATHS".format(pmf, r), "CR-{}{}".format(pmf, s), "MCP-{}{}".format(pmf, s), "EV-{}{}".format(pmf, s),
                        "UCB-{}{}".format(pmf, s)]  
    writer.writerow(header)

    for dname in os.listdir(dag_path):   
        # print(dname)
        
        # Load the DAG topology.
        with open('{}/{}'.format(dag_path, dname), 'rb') as file:
            T = dill.load(file)
        # Convert to DAG object.
        G = DAG(T)   
        
        for q, rtask, rmach, muccr, v in product(nprocessors, Rs, Rs, ccrs, Vs):
            for run in range(runs):   
                data = [dname[:-5], q, rtask, rmach, muccr, v, run]
                
                # Set costs.
                params = (rtask, rmach, 1.0, v)
                G.set_random_weights(nprocessors=q, comp_method="CNB", comp_params=params, vband=v, muccr=muccr, vccr=v)            
                           
                # Compute actual CCR.
                act_ccr = G.ccr()
                data.append(act_ccr)
                
                # Compute minimal serial time.
                mst = G.minimal_serial_time()  
                data.append(mst)
                    
                # Compute makespan lower bound.
                lb = G.makespan_lower_bound()
                data.append(lb)
                
                # Compute makespan for random selection policy.
                rand_prios = {t : G.size - i for i, t in enumerate(G.top_sort)}
                mkspan = priority_scheduling(G, priorities=rand_prios)
                data.append(mkspan)
                
                # Equivalent for CPOP.
                start = timer()
                cpop_prios = G.get_upward_ranks(avg_type="M") 
                elapsed = timer() - start
                data.append(elapsed)
                
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
                data.append(mkspan)
                            
                # =============================================================================
                # Averages.   
                # =============================================================================
                for avg in avgs: 
                    # HEFT.
                    U = G.get_upward_ranks(avg_type=avg)
                    heft_mkspan = priority_scheduling(G, priorities=U)
                    data.append(heft_mkspan)
                    
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
                    data.append(cpop_mkspan)
    
                # =============================================================================
                # Critical path bounds.           
                # =============================================================================
                
                opt, opt_path = G.optimistic_critical_path(return_path=True)
                opt_ranks = {t : min(opt[t].values()) for t in G.top_sort}
                heft_mkspan = priority_scheduling(G, priorities=opt_ranks)
                data.append(heft_mkspan)                
                cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=opt_path)        
                data.append(cpop_mkspan)
                
                # Pessimistic critical paths.
                pes, pes_path = G.optimistic_critical_path(pessimistic=True, return_path=True)
                pes_ranks = {t : max(opt[t].values()) for t in G.top_sort}
                heft_mkspan = priority_scheduling(G, priorities=pes_ranks)
                data.append(heft_mkspan)
                cpop_mkspan = cpop(G, priorities=cpop_prios, critical_path=pes_path)   
                data.append(cpop_mkspan)
                          
                # =============================================================================
                # Monte Carlo.           
                # =============================================================================
                for pmf in ["A", "H"]: 
                    for r in [10, 1000]: # Do 10 new realizations rather than just sampling 10 in order to time etc.
                    
                        L, path_counts, criticalities, timings = G.monte_carlo(realizations=r, pmf=pmf, times=True) 
                        
                        for k, val in timings.items():
                            data.append(val)
                        
                        # Number of critical paths.
                        paths = len(path_counts)
                        data.append(paths)
                        
                        # Criticalities.
                        mkspan = priority_scheduling(G, priorities=criticalities)
                        data.append(mkspan)
                        
                        # CPOP (most frequently critical path). 
                        mcp = max(path_counts, key=path_counts.get)
                        mkspan = cpop(G, priorities=cpop_prios, critical_path=mcp) 
                        data.append(mkspan)
                        
                        # Mean.
                        means = {t: np.mean(L[t]) for t in G.top_sort}
                        mkspan = priority_scheduling(G, priorities=means) 
                        data.append(mkspan)
                        
                        # UCB.
                        ucbs = {t: np.mean(L[t]) + np.std(L[t]) for t in G.top_sort}
                        mkspan = priority_scheduling(G, priorities=ucbs)
                        data.append(mkspan)  
                # Write row to file.
                writer.writerow(data)


    
    

