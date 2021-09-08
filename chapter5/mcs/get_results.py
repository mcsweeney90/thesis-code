#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the results presented in Section 5.3 (Accelerating MCS).
NOTE: to avoid overwriting the data 'mcs.csv' that was used in thesis, have changed the name of save destination to 'new_mcs.csv'. 
"""

import os, dill, random, scipy.stats
import numpy as np
import pandas as pd
from math import sqrt

import sys
sys.path.append("../")
from src import StochTaskDAG, SSTAR, SDLS, HEFT

def _broadcast_concatenate(x, y, axis):
    """
    Broadcast then concatenate arrays, leaving concatenation axis last.
    Taken from Scipy implementation of mannwhitneyu.
    Helper function for below.
    """
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    z = np.broadcast(x[..., 0], y[..., 0])
    x = np.broadcast_to(x, z.shape + (x.shape[-1],))
    y = np.broadcast_to(y, z.shape + (y.shape[-1],))
    z = np.concatenate((x, y), axis=-1)
    return x, y, z

def prob_less(A, B):
    """
    Rough estimate of the probability that empirical distribution A is less than empirical distribution B.
    Taken from Scipy implementation of mannwhitneyu.
    """
    x, y, xy = _broadcast_concatenate(A, B, 0)
    n1, n2 = x.shape[-1], y.shape[-1]
    ranks = scipy.stats.rankdata(xy, axis=-1)  # method 2, step 1
    R1 = ranks[..., :n1].sum(axis=-1)    # method 2, step 2
    U1 = R1 - n1*(n1+1)/2                # method 2, step 3
    return (n1 * n2 - U1)/(len(A) * len(B)) 

# Lambda functions.
real_func = lambda r : abs(random.uniform(r.mu - sqrt(3)*r.sd, r.mu + sqrt(3)*r.sd))
s_func = lambda r : r.mu + r.sd if r.mu > r.sd else r.mu + (r.sd/r.mu)

# =============================================================================
# Common variables.
# =============================================================================    

size = 100
stg_dag_path = '../../graphs/STG/{}'.format(size)
mucovs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5] 
runs = 10
trials = 100

# =============================================================================
# Generate results.
# =============================================================================

data = []
Cs = np.random.uniform(0, 3, trials)
for dname in os.listdir(stg_dag_path): 
    print("\nDAG: {}".format(dname[:-5]))
    with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
        T = dill.load(file)
    G = StochTaskDAG(T)
    
    for mucov in mucovs: 
        for run in range(runs):
            G.set_weights(mucov=mucov) 
            graph_data = {"DAG" : dname[:-5], "COV" : mucov, "RUN" : run}  
                    
            # HEFT.
            heft_schedule = SSTAR(G, det_heuristic=HEFT)
            heft = heft_schedule.longest_path(method="MC", mc_samples=1000)    
            heft_mu = np.mean(heft)
            graph_data["HEFT MU"] = heft_mu
            heft_sd = np.std(heft)
            graph_data["HEFT SD"] = heft_sd
            
            # SHEFT.
            sheft_schedule = SSTAR(G, det_heuristic=HEFT, scal_func=s_func)
            sheft = sheft_schedule.longest_path(method="MC", mc_samples=1000)
            sheft_mu = np.mean(sheft)
            graph_data["SHEFT MU"] = sheft_mu
            sheft_sd = np.std(sheft)
            graph_data["SHEFT SD"] = sheft_sd
            prob = prob_less(sheft, heft)
            graph_data["SHEFT PROB"] = prob        
            
            # SDLS.
            sdls_schedule = SDLS(G, insertion=None)
            sdls = sdls_schedule.longest_path(method="MC", mc_samples=1000)  
            sdls_mu = np.mean(sdls)
            graph_data["SDLS MU"] = sdls_mu
            sdls_sd = np.std(sdls)
            graph_data["SDLS SD"] = sdls_sd
            prob = prob_less(sdls, heft)
            graph_data["SDLS PROB"] = prob
            
            # MCS approach.
            mu1, sd1, prob1 = float("inf"), float("inf"), 0
            for _ in range(trials):
                sched = SSTAR(G, det_heuristic=HEFT, scal_func=real_func)
                mkspan = sched.longest_path(method="MC", mc_samples=1000)
                mu1 = min(mu1, np.mean(mkspan))
                sd1 = min(sd1, np.std(mkspan))
                p = prob_less(mkspan, heft)
                prob1 = max(prob1, p)
            graph_data["MCS{} MU".format(trials)] = mu1   
            graph_data["MCS{} SD".format(trials)] = sd1
            graph_data["MCS{} PROB".format(trials)] = prob1
            
            mu2, sd2, prob2 = float("inf"), float("inf"), 0
            for i in range(trials):
                s_func = lambda r : r.mu + Cs[i]*r.sd 
                sched = SSTAR(G, det_heuristic=HEFT, scal_func=s_func)
                mkspan = sched.longest_path(method="MC", mc_samples=1000)
                mu2 = min(mu2, np.mean(mkspan))
                sd2 = min(sd2, np.std(mkspan))
                p = prob_less(mkspan, heft)
                prob2 = max(prob2, p)
            graph_data["UCB{} MU".format(trials)] = mu2   
            graph_data["UCB{} SD".format(trials)] = sd2
            graph_data["UCB{} PROB".format(trials)] = prob2
            
            data.append(graph_data)  

# Save data. 
df = pd.DataFrame(data)  
df.to_csv('new_mcs.csv', encoding='utf-8', index=False)      