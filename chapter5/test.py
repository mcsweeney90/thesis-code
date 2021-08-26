#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script.
TODO: RobHEFT and MCS need serious look. SDLS wouldn't hurt either but think that's probably fine.
"""

import os, dill, random, scipy.stats
import numpy as np
from timeit import default_timer as timer
from functools import partial
from math import sqrt
from src import RV, TDAG, SSTAR, SDLS, RobHEFT, HEFT, MCS, PEFT


def _broadcast_concatenate(x, y, axis):
    """
    Broadcast then concatenate arrays, leaving concatenation axis last.
    Taken from Scipy implementation of mannwhitneyu.
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

real_func = lambda r : abs(random.uniform(r.mu - sqrt(3)*r.sd, r.mu + sqrt(3)*r.sd))

size = 100
stg_dag_path = '../graphs/STG/{}'.format(size)

# =============================================================================
# Full set.
# =============================================================================

trials = 10

wins = {"MU" : {"S":0, "MC":0}, "SD" : {"S":0, "MC":0}, "PROB" : {"S":0, "MC":0}}
Cs = np.random.uniform(0, 3, trials)

for dname in os.listdir(stg_dag_path): 
    print("\nDAG: {}".format(dname[:-5]))
    with open('{}/{}'.format(stg_dag_path, dname), 'rb') as file:
        T = dill.load(file)
    G = TDAG(T)
    G.set_weights(mucov=0.2) 
    
    # HEFT.
    heft_schedule = SSTAR(G, det_heuristic=HEFT)
    heft = heft_schedule.longest_path(method="MC", mc_samples=1000) 
    print(np.mean(heft), np.std(heft))
    
    mu1, sd1, prob1 = float("inf"), float("inf"), 0
    for _ in range(trials):
        sched = SSTAR(G, det_heuristic=HEFT, scal_func=real_func)
        mkspan = sched.longest_path(method="MC", mc_samples=1000)
        mu1 = min(mu1, np.mean(mkspan))
        sd1 = min(sd1, np.std(mkspan))
        p = prob_less(mkspan, heft)
        prob1 = max(prob1, p)
    print(mu1, sd1, prob1)
    
    mu2, sd2, prob2 = float("inf"), float("inf"), 0
    for i in range(trials):
        s_func = lambda r : r.mu + Cs[i]*r.sd 
        sched = SSTAR(G, det_heuristic=HEFT, scal_func=s_func)
        mkspan = sched.longest_path(method="MC", mc_samples=1000)
        mu2 = min(mu2, np.mean(mkspan))
        sd2 = min(sd2, np.std(mkspan))
        p = prob_less(mkspan, heft)
        prob2 = max(prob2, p)
    print(mu2, sd2, prob2)
    
    if prob1 < prob2:
        wins["PROB"]["S"] += 1
    else:
        wins["PROB"]["MC"] += 1
    
    if mu1 < mu2:
        wins["MU"]["MC"] += 1
    else:
        wins["MU"]["S"] += 1
        
    if sd1 < sd2:
        wins["SD"]["MC"] += 1
    else:
        wins["SD"]["S"] += 1

print(wins)
        
    
    # # SHEFT approach.
    # idx = -1
    # for i, c in enumerate(Cs):
    #     s_func = lambda r : r.mu + c*r.sd 
    #     sched = SSTAR(G, det_heuristic=HEFT, scal_func=s_func)
    #     mkspan = sched.longest_path(method="MC", mc_samples=1000)
    #     mu = np.mean(mkspan)
    #     prob = prob_less(mkspan, heft)
    #     if prob > 0.5:
    #         idx = i
    #         break
    # if idx < 0:
    #     ucb_trials.append(max_trials)
    # else:
    #     ucb_trials.append(idx)
    
    # # Now see if can find a corresponding schedule in the same number of iterations using MCS approach.
    # mcs_runs = max_trials if idx < 0 else idx + 1    
    # win = False
    # for j in range(mcs_runs):
    #     sched = SSTAR(G, det_heuristic=HEFT, scal_func=real_func)
    #     mkspan = sched.longest_path(method="MC", mc_samples=1000)
    #     prob = prob_less(mkspan, heft)
    #     if prob > 0.5:
    #         win = True
    #         break
    # mcs_wins.append(win)
               
    
    # # SHEFT.
    # s_func = lambda r : r.mu + r.sd if r.mu > r.sd else r.mu + (r.sd/r.mu)
    # sheft_schedule = SSTAR(G, det_heuristic=HEFT, scal_func=s_func)
    # sheft = sheft_schedule.longest_path(method="MC", mc_samples=1000)
    
    # # SDLS.
    # sdls_schedule = SDLS(G, insertion=None)
    # sdls = sdls_schedule.longest_path(method="MC", mc_samples=1000)       
    
    # # Compute probability that SHEFT better than HEFT.
    # prob = prob_less(sheft, heft) 
    # probs["SHEFT"].append(prob)
    
    # # Compute probability that SDLS better than HEFT.
    # prob = prob_less(sdls, heft)
    # probs["SDLS"].append(prob)
   


# =============================================================================
# Single graph (useful for debugging).
# =============================================================================
    

# with open('{}/147.dill'.format(stg_dag_path), 'rb') as file:
#     T = dill.load(file)
# G = TDAG(T)
# G.set_weights(mucov=0.1)

# # HEFT.
# heft_schedule = SSTAR(G, det_heuristic=HEFT)
# heft = heft_schedule.longest_path(method="MC", mc_samples=1000)

# # SHEFT.
# s_func = lambda r : r.mu + r.sd if r.mu > r.sd else r.mu + (r.sd/r.mu)
# sheft_schedule = SSTAR(G, det_heuristic=HEFT, scal_func=s_func)
# sheft = sheft_schedule.longest_path(method="MC", mc_samples=1000)

# # SDLS.
# sdls_schedule = SDLS(G, insertion=None)
# sdls = sdls_schedule.longest_path(method="MC", mc_samples=1000)  

# trials = 10
# Cs = np.random.uniform(0, 3, trials)
# for i in range(trials):
#     c = Cs[i]
#     s_func = lambda r : r.mu + c*r.sd 
#     sched = SSTAR(G, det_heuristic=HEFT, scal_func=s_func)
#     mkspan = sched.longest_path(method="MC", mc_samples=1000)
#     prob = prob_less(mkspan, heft)
#     if prob > 0.5:
#         print(c, i)
#         break
    



